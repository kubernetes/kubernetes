/*
Copyright 2014 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package cmd

import (
	"bytes"
	"fmt"
	"io"

	jsonpatch "github.com/evanphx/json-patch"
	"github.com/golang/glog"
	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/json"

	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
	"k8s.io/kubernetes/pkg/printers"
)

// AnnotateOptions have the data required to perform the annotate operation
type AnnotateOptions struct {
	// Filename options
	resource.FilenameOptions

	// Common user flags
	overwrite         bool
	local             bool
	dryrun            bool
	all               bool
	resourceVersion   string
	selector          string
	outputFormat      string
	recordChangeCause bool

	// results of arg parsing
	resources         []string
	newAnnotations    map[string]string
	removeAnnotations []string

	// Common share fields
	out io.Writer
}

var (
	annotateLong = templates.LongDesc(`
		Update the annotations on one or more resources

		All Kubernetes objects support the ability to store additional data with the object as
		annotations. Annotations are key/value pairs that can be larger than labels and include
		arbitrary string values such as structured JSON. Tools and system extensions may use
		annotations to store their own data.  

		Attempting to set an annotation that already exists will fail unless --overwrite is set.
		If --resource-version is specified and does not match the current resource version on
		the server the command will fail.`)

	annotateExample = templates.Examples(i18n.T(`
    # Update pod 'foo' with the annotation 'description' and the value 'my frontend'.
    # If the same annotation is set multiple times, only the last value will be applied
    kubectl annotate pods foo description='my frontend'

    # Update a pod identified by type and name in "pod.json"
    kubectl annotate -f pod.json description='my frontend'

    # Update pod 'foo' with the annotation 'description' and the value 'my frontend running nginx', overwriting any existing value.
    kubectl annotate --overwrite pods foo description='my frontend running nginx'

    # Update all pods in the namespace
    kubectl annotate pods --all description='my frontend running nginx'

    # Update pod 'foo' only if the resource is unchanged from version 1.
    kubectl annotate pods foo description='my frontend running nginx' --resource-version=1

    # Update pod 'foo' by removing an annotation named 'description' if it exists.
    # Does not require the --overwrite flag.
    kubectl annotate pods foo description-`))
)

func NewCmdAnnotate(f cmdutil.Factory, out io.Writer) *cobra.Command {
	options := &AnnotateOptions{}

	// retrieve a list of handled resources from printer as valid args
	validArgs, argAliases := []string{}, []string{}
	p, err := f.Printer(nil, printers.PrintOptions{
		ColumnLabels: []string{},
	})
	cmdutil.CheckErr(err)
	if p != nil {
		validArgs = p.HandledResources()
		argAliases = kubectl.ResourceAliases(validArgs)
	}

	cmd := &cobra.Command{
		Use:     "annotate [--overwrite] (-f FILENAME | TYPE NAME) KEY_1=VAL_1 ... KEY_N=VAL_N [--resource-version=version]",
		Short:   i18n.T("Update the annotations on a resource"),
		Long:    annotateLong + "\n\n" + cmdutil.ValidResourceTypeList(f),
		Example: annotateExample,
		Run: func(cmd *cobra.Command, args []string) {
			if err := options.Complete(out, cmd, args); err != nil {
				cmdutil.CheckErr(cmdutil.UsageErrorf(cmd, "%v", err))
			}
			if err := options.Validate(); err != nil {
				cmdutil.CheckErr(cmdutil.UsageErrorf(cmd, "%v", err))
			}
			cmdutil.CheckErr(options.RunAnnotate(f, cmd))
		},
		ValidArgs:  validArgs,
		ArgAliases: argAliases,
	}
	cmdutil.AddPrinterFlags(cmd)
	cmdutil.AddIncludeUninitializedFlag(cmd)
	cmd.Flags().Bool("overwrite", false, "If true, allow annotations to be overwritten, otherwise reject annotation updates that overwrite existing annotations.")
	cmd.Flags().Bool("local", false, "If true, annotation will NOT contact api-server but run locally.")
	cmd.Flags().StringP("selector", "l", "", "Selector (label query) to filter on, not including uninitialized ones, supports '=', '==', and '!='.(e.g. -l key1=value1,key2=value2).")
	cmd.Flags().Bool("all", false, "Select all resources, including uninitialized ones, in the namespace of the specified resource types.")
	cmd.Flags().String("resource-version", "", i18n.T("If non-empty, the annotation update will only succeed if this is the current resource-version for the object. Only valid when specifying a single resource."))
	usage := "identifying the resource to update the annotation"
	cmdutil.AddFilenameOptionFlags(cmd, &options.FilenameOptions, usage)
	cmdutil.AddDryRunFlag(cmd)
	cmdutil.AddRecordFlag(cmd)
	cmdutil.AddInclude3rdPartyFlags(cmd)

	return cmd
}

// Complete adapts from the command line args and factory to the data required.
func (o *AnnotateOptions) Complete(out io.Writer, cmd *cobra.Command, args []string) (err error) {
	o.out = out
	o.local = cmdutil.GetFlagBool(cmd, "local")
	o.overwrite = cmdutil.GetFlagBool(cmd, "overwrite")
	o.all = cmdutil.GetFlagBool(cmd, "all")
	o.resourceVersion = cmdutil.GetFlagString(cmd, "resource-version")
	o.selector = cmdutil.GetFlagString(cmd, "selector")
	o.outputFormat = cmdutil.GetFlagString(cmd, "output")
	o.dryrun = cmdutil.GetDryRunFlag(cmd)
	o.recordChangeCause = cmdutil.GetRecordFlag(cmd)

	// retrieves resource and annotation args from args
	// also checks args to verify that all resources are specified before annotations
	resources, annotationArgs, err := cmdutil.GetResourcesAndPairs(args, "annotation")
	if err != nil {
		return err
	}
	o.resources = resources
	o.newAnnotations, o.removeAnnotations, err = parseAnnotations(annotationArgs)
	return err
}

// Validate checks to the AnnotateOptions to see if there is sufficient information run the command.
func (o AnnotateOptions) Validate() error {
	if len(o.resources) < 1 && cmdutil.IsFilenameSliceEmpty(o.Filenames) {
		return fmt.Errorf("one or more resources must be specified as <resource> <name> or <resource>/<name>")
	}
	if len(o.newAnnotations) < 1 && len(o.removeAnnotations) < 1 {
		return fmt.Errorf("at least one annotation update is required")
	}
	return validateAnnotations(o.removeAnnotations, o.newAnnotations)
}

// RunAnnotate does the work
func (o AnnotateOptions) RunAnnotate(f cmdutil.Factory, cmd *cobra.Command) error {
	namespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	changeCause := f.Command(cmd, false)

	includeUninitialized := cmdutil.ShouldIncludeUninitialized(cmd, false)

	var b *resource.Builder
	if o.local {
		b = f.NewBuilder().
			Local(f.ClientForMapping)
	} else {
		b = f.NewUnstructuredBuilder().
			LabelSelectorParam(o.selector).
			ResourceTypeOrNameArgs(o.all, o.resources...).
			Latest()
	}
	r := b.ContinueOnError().
		NamespaceParam(namespace).DefaultNamespace().
		FilenameParam(enforceNamespace, &o.FilenameOptions).
		IncludeUninitialized(includeUninitialized).
		Flatten().
		Do()
	if err := r.Err(); err != nil {
		return err
	}

	var singleItemImpliedResource bool
	r.IntoSingleItemImplied(&singleItemImpliedResource)

	// only apply resource version locking on a single resource.
	// we must perform this check after o.builder.Do() as
	// []o.resources can not not accurately return the proper number
	// of resources when they are not passed in "resource/name" format.
	if !singleItemImpliedResource && len(o.resourceVersion) > 0 {
		return fmt.Errorf("--resource-version may only be used with a single resource")
	}

	return r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}

		var outputObj runtime.Object
		var obj runtime.Object

		obj, err = info.Mapping.ConvertToVersion(info.Object, info.Mapping.GroupVersionKind.GroupVersion())
		if err != nil {
			return err
		}

		if o.dryrun || o.local {
			if err := o.updateAnnotations(obj); err != nil {
				return err
			}
			outputObj = obj
		} else {
			name, namespace := info.Name, info.Namespace
			oldData, err := json.Marshal(obj)
			if err != nil {
				return err
			}
			// If we should record change-cause, add it to new annotations
			if cmdutil.ContainsChangeCause(info) || o.recordChangeCause {
				o.newAnnotations[kubectl.ChangeCauseAnnotation] = changeCause
			}
			if err := o.updateAnnotations(obj); err != nil {
				return err
			}
			newData, err := json.Marshal(obj)
			if err != nil {
				return err
			}
			patchBytes, err := jsonpatch.CreateMergePatch(oldData, newData)
			createdPatch := err == nil
			if err != nil {
				glog.V(2).Infof("couldn't compute patch: %v", err)
			}

			mapping := info.ResourceMapping()
			client, err := f.UnstructuredClientForMapping(mapping)
			if err != nil {
				return err
			}
			helper := resource.NewHelper(client, mapping)

			if createdPatch {
				outputObj, err = helper.Patch(namespace, name, types.MergePatchType, patchBytes)
			} else {
				outputObj, err = helper.Replace(namespace, name, false, obj)
			}
			if err != nil {
				return err
			}
		}

		var mapper meta.RESTMapper
		if o.local {
			mapper, _ = f.Object()
		} else {
			mapper, _, err = f.UnstructuredObject()
			if err != nil {
				return err
			}
		}
		if len(o.outputFormat) > 0 {
			return f.PrintObject(cmd, o.local, mapper, outputObj, o.out)
		}
		cmdutil.PrintSuccess(mapper, false, o.out, info.Mapping.Resource, info.Name, o.dryrun, "annotated")
		return nil
	})
}

// parseAnnotations retrieves new and remove annotations from annotation args
func parseAnnotations(annotationArgs []string) (map[string]string, []string, error) {
	return cmdutil.ParsePairs(annotationArgs, "annotation", true)
}

// validateAnnotations checks the format of annotation args and checks removed annotations aren't in the new annotations map
func validateAnnotations(removeAnnotations []string, newAnnotations map[string]string) error {
	var modifyRemoveBuf bytes.Buffer
	for _, removeAnnotation := range removeAnnotations {
		if _, found := newAnnotations[removeAnnotation]; found {
			if modifyRemoveBuf.Len() > 0 {
				modifyRemoveBuf.WriteString(", ")
			}
			modifyRemoveBuf.WriteString(fmt.Sprintf(removeAnnotation))
		}
	}
	if modifyRemoveBuf.Len() > 0 {
		return fmt.Errorf("can not both modify and remove the following annotation(s) in the same command: %s", modifyRemoveBuf.String())
	}

	return nil
}

// validateNoAnnotationOverwrites validates that when overwrite is false, to-be-updated annotations don't exist in the object annotation map (yet)
func validateNoAnnotationOverwrites(accessor metav1.Object, annotations map[string]string) error {
	var buf bytes.Buffer
	for key := range annotations {
		// change-cause annotation can always be overwritten
		if key == kubectl.ChangeCauseAnnotation {
			continue
		}
		if value, found := accessor.GetAnnotations()[key]; found {
			if buf.Len() > 0 {
				buf.WriteString("; ")
			}
			buf.WriteString(fmt.Sprintf("'%s' already has a value (%s)", key, value))
		}
	}
	if buf.Len() > 0 {
		return fmt.Errorf("--overwrite is false but found the following declared annotation(s): %s", buf.String())
	}
	return nil
}

// updateAnnotations updates annotations of obj
func (o AnnotateOptions) updateAnnotations(obj runtime.Object) error {
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return err
	}
	if !o.overwrite {
		if err := validateNoAnnotationOverwrites(accessor, o.newAnnotations); err != nil {
			return err
		}
	}

	annotations := accessor.GetAnnotations()
	if annotations == nil {
		annotations = make(map[string]string)
	}

	for key, value := range o.newAnnotations {
		annotations[key] = value
	}
	for _, annotation := range o.removeAnnotations {
		delete(annotations, annotation)
	}
	accessor.SetAnnotations(annotations)

	if len(o.resourceVersion) != 0 {
		accessor.SetResourceVersion(o.resourceVersion)
	}
	return nil
}
