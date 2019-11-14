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

package annotate

import (
	"bytes"
	"fmt"
	"io"

	jsonpatch "github.com/evanphx/json-patch"
	"github.com/spf13/cobra"
	"k8s.io/klog"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/json"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/printers"
	"k8s.io/cli-runtime/pkg/resource"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/polymorphichelpers"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

// AnnotateOptions have the data required to perform the annotate operation
type AnnotateOptions struct {
	PrintFlags *genericclioptions.PrintFlags
	PrintObj   printers.ResourcePrinterFunc

	// Filename options
	resource.FilenameOptions
	RecordFlags *genericclioptions.RecordFlags

	// Common user flags
	overwrite       bool
	local           bool
	dryrun          bool
	all             bool
	resourceVersion string
	selector        string
	fieldSelector   string
	outputFormat    string

	// results of arg parsing
	resources                    []string
	newAnnotations               map[string]string
	removeAnnotations            []string
	Recorder                     genericclioptions.Recorder
	namespace                    string
	enforceNamespace             bool
	builder                      *resource.Builder
	unstructuredClientForMapping func(mapping *meta.RESTMapping) (resource.RESTClient, error)

	genericclioptions.IOStreams
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

// NewAnnotateOptions creates the options for annotate
func NewAnnotateOptions(ioStreams genericclioptions.IOStreams) *AnnotateOptions {
	return &AnnotateOptions{
		PrintFlags: genericclioptions.NewPrintFlags("annotated").WithTypeSetter(scheme.Scheme),

		RecordFlags: genericclioptions.NewRecordFlags(),
		Recorder:    genericclioptions.NoopRecorder{},
		IOStreams:   ioStreams,
	}
}

// NewCmdAnnotate creates the `annotate` command
func NewCmdAnnotate(parent string, f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *cobra.Command {
	o := NewAnnotateOptions(ioStreams)

	cmd := &cobra.Command{
		Use:                   "annotate [--overwrite] (-f FILENAME | TYPE NAME) KEY_1=VAL_1 ... KEY_N=VAL_N [--resource-version=version]",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Update the annotations on a resource"),
		Long:                  annotateLong + "\n\n" + cmdutil.SuggestAPIResources(parent),
		Example:               annotateExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.RunAnnotate())
		},
	}

	// bind flag structs
	o.RecordFlags.AddFlags(cmd)
	o.PrintFlags.AddFlags(cmd)

	cmd.Flags().BoolVar(&o.overwrite, "overwrite", o.overwrite, "If true, allow annotations to be overwritten, otherwise reject annotation updates that overwrite existing annotations.")
	cmd.Flags().BoolVar(&o.local, "local", o.local, "If true, annotation will NOT contact api-server but run locally.")
	cmd.Flags().StringVarP(&o.selector, "selector", "l", o.selector, "Selector (label query) to filter on, not including uninitialized ones, supports '=', '==', and '!='.(e.g. -l key1=value1,key2=value2).")
	cmd.Flags().StringVar(&o.fieldSelector, "field-selector", o.fieldSelector, "Selector (field query) to filter on, supports '=', '==', and '!='.(e.g. --field-selector key1=value1,key2=value2). The server only supports a limited number of field queries per type.")
	cmd.Flags().BoolVar(&o.all, "all", o.all, "Select all resources, including uninitialized ones, in the namespace of the specified resource types.")
	cmd.Flags().StringVar(&o.resourceVersion, "resource-version", o.resourceVersion, i18n.T("If non-empty, the annotation update will only succeed if this is the current resource-version for the object. Only valid when specifying a single resource."))
	usage := "identifying the resource to update the annotation"
	cmdutil.AddFilenameOptionFlags(cmd, &o.FilenameOptions, usage)
	cmdutil.AddDryRunFlag(cmd)

	return cmd
}

// Complete adapts from the command line args and factory to the data required.
func (o *AnnotateOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	var err error

	o.RecordFlags.Complete(cmd)
	o.Recorder, err = o.RecordFlags.ToRecorder()
	if err != nil {
		return err
	}

	o.outputFormat = cmdutil.GetFlagString(cmd, "output")
	o.dryrun = cmdutil.GetDryRunFlag(cmd)

	if o.dryrun {
		o.PrintFlags.Complete("%s (dry run)")
	}
	printer, err := o.PrintFlags.ToPrinter()
	if err != nil {
		return err
	}
	o.PrintObj = func(obj runtime.Object, out io.Writer) error {
		return printer.PrintObj(obj, out)
	}

	o.namespace, o.enforceNamespace, err = f.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return err
	}
	o.builder = f.NewBuilder()
	o.unstructuredClientForMapping = f.UnstructuredClientForMapping

	// retrieves resource and annotation args from args
	// also checks args to verify that all resources are specified before annotations
	resources, annotationArgs, err := cmdutil.GetResourcesAndPairs(args, "annotation")
	if err != nil {
		return err
	}
	o.resources = resources
	o.newAnnotations, o.removeAnnotations, err = parseAnnotations(annotationArgs)
	if err != nil {
		return err
	}

	return nil
}

// Validate checks to the AnnotateOptions to see if there is sufficient information run the command.
func (o AnnotateOptions) Validate() error {
	if o.all && len(o.selector) > 0 {
		return fmt.Errorf("cannot set --all and --selector at the same time")
	}
	if o.all && len(o.fieldSelector) > 0 {
		return fmt.Errorf("cannot set --all and --field-selector at the same time")
	}
	if len(o.resources) < 1 && cmdutil.IsFilenameSliceEmpty(o.Filenames, o.Kustomize) {
		return fmt.Errorf("one or more resources must be specified as <resource> <name> or <resource>/<name>")
	}
	if len(o.newAnnotations) < 1 && len(o.removeAnnotations) < 1 {
		return fmt.Errorf("at least one annotation update is required")
	}
	return validateAnnotations(o.removeAnnotations, o.newAnnotations)
}

// RunAnnotate does the work
func (o AnnotateOptions) RunAnnotate() error {
	b := o.builder.
		Unstructured().
		LocalParam(o.local).
		ContinueOnError().
		NamespaceParam(o.namespace).DefaultNamespace().
		FilenameParam(o.enforceNamespace, &o.FilenameOptions).
		Flatten()

	if !o.local {
		b = b.LabelSelectorParam(o.selector).
			FieldSelectorParam(o.fieldSelector).
			ResourceTypeOrNameArgs(o.all, o.resources...).
			Latest()
	}

	r := b.Do()
	if err := r.Err(); err != nil {
		return err
	}

	var singleItemImpliedResource bool
	r.IntoSingleItemImplied(&singleItemImpliedResource)

	// only apply resource version locking on a single resource.
	// we must perform this check after o.builder.Do() as
	// []o.resources can not accurately return the proper number
	// of resources when they are not passed in "resource/name" format.
	if !singleItemImpliedResource && len(o.resourceVersion) > 0 {
		return fmt.Errorf("--resource-version may only be used with a single resource")
	}

	return r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}

		var outputObj runtime.Object
		obj := info.Object

		if o.dryrun || o.local {
			if err := o.updateAnnotations(obj); err != nil {
				return err
			}
			outputObj = obj
		} else {
			name, namespace := info.Name, info.Namespace

			if len(o.resourceVersion) != 0 {
				// ensure resourceVersion is always sent in the patch by clearing it from the starting JSON
				accessor, err := meta.Accessor(obj)
				if err != nil {
					return err
				}
				accessor.SetResourceVersion("")
			}

			oldData, err := json.Marshal(obj)
			if err != nil {
				return err
			}
			if err := o.Recorder.Record(info.Object); err != nil {
				klog.V(4).Infof("error recording current command: %v", err)
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
				klog.V(2).Infof("couldn't compute patch: %v", err)
			}

			mapping := info.ResourceMapping()
			client, err := o.unstructuredClientForMapping(mapping)
			if err != nil {
				return err
			}
			helper := resource.NewHelper(client, mapping)

			if createdPatch {
				outputObj, err = helper.Patch(namespace, name, types.MergePatchType, patchBytes, nil)
			} else {
				outputObj, err = helper.Replace(namespace, name, false, obj)
			}
			if err != nil {
				return err
			}
		}

		return o.PrintObj(outputObj, o.Out)
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
		if key == polymorphichelpers.ChangeCauseAnnotation {
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
