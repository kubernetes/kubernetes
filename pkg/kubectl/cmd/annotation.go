/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"strings"

	"github.com/spf13/cobra"
	"k8s.io/kubernetes/pkg/api"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/runtime"
)

// AnnotateOptions have the data required to perform the annotate operation
type AnnotateOptions struct {
	out               io.Writer
	resources         []string
	newAnnotations    map[string]string
	removeAnnotations []string
	builder           *resource.Builder

	overwrite       bool
	all             bool
	resourceVersion string
}

const (
	annotate_long = `Update the annotations on one or more resources.

An annotation is a key/value pair that can hold larger (compared to a label), and possibly not human-readable, data.
It is intended to store non-identifying auxiliary data, especially data manipulated by tools and system extensions.
If --overwrite is true, then existing annotations can be overwritten, otherwise attempting to overwrite an annotation will result in an error.
If --resource-version is specified, then updates will use this resource version, otherwise the existing resource-version will be used.

Possible resources include (case insensitive): pods (po), services (svc),
replicationcontrollers (rc), nodes (no), events (ev), componentstatuses (cs),
limitranges (limits), persistentvolumes (pv), persistentvolumeclaims (pvc),
resourcequotas (quota) or secrets.`
	annotate_example = `# Update pod 'foo' with the annotation 'description' and the value 'my frontend'.
# If the same annotation is set multiple times, only the last value will be applied
$ kubectl annotate pods foo description='my frontend'

# Update pod 'foo' with the annotation 'description' and the value 'my frontend running nginx', overwriting any existing value.
$ kubectl annotate --overwrite pods foo description='my frontend running nginx'

# Update all pods in the namespace
$ kubectl annotate pods --all description='my frontend running nginx'

# Update pod 'foo' only if the resource is unchanged from version 1.
$ kubectl annotate pods foo description='my frontend running nginx' --resource-version=1

# Update pod 'foo' by removing an annotation named 'description' if it exists.
# Does not require the --overwrite flag.
$ kubectl annotate pods foo description-`
)

func NewCmdAnnotate(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	options := &AnnotateOptions{}

	cmd := &cobra.Command{
		Use:     "annotate [--overwrite] RESOURCE NAME KEY_1=VAL_1 ... KEY_N=VAL_N [--resource-version=version]",
		Short:   "Update the annotations on a resource",
		Long:    annotate_long,
		Example: annotate_example,
		Run: func(cmd *cobra.Command, args []string) {
			if err := options.Complete(f, args, out); err != nil {
				cmdutil.CheckErr(err)
			}
			if err := options.Validate(args); err != nil {
				cmdutil.CheckErr(cmdutil.UsageError(cmd, err.Error()))
			}
			if err := options.RunAnnotate(); err != nil {
				cmdutil.CheckErr(err)
			}
		},
	}
	cmdutil.AddPrinterFlags(cmd)
	cmd.Flags().BoolVar(&options.overwrite, "overwrite", false, "If true, allow annotations to be overwritten, otherwise reject annotation updates that overwrite existing annotations.")
	cmd.Flags().BoolVar(&options.all, "all", false, "select all resources in the namespace of the specified resource types")
	cmd.Flags().StringVar(&options.resourceVersion, "resource-version", "", "If non-empty, the annotation update will only succeed if this is the current resource-version for the object. Only valid when specifying a single resource.")
	return cmd
}

// Complete adapts from the command line args and factory to the data required.
func (o *AnnotateOptions) Complete(f *cmdutil.Factory, args []string, out io.Writer) (err error) {
	namespace, _, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	// retrieves resource and annotation args from args
	// also checks args to verify that all resources are specified before annotations
	annotationArgs := []string{}
	metAnnotaionArg := false
	for _, s := range args {
		isAnnotation := strings.Contains(s, "=") || strings.HasSuffix(s, "-")
		switch {
		case !metAnnotaionArg && isAnnotation:
			metAnnotaionArg = true
			fallthrough
		case metAnnotaionArg && isAnnotation:
			annotationArgs = append(annotationArgs, s)
		case !metAnnotaionArg && !isAnnotation:
			o.resources = append(o.resources, s)
		case metAnnotaionArg && !isAnnotation:
			return fmt.Errorf("all resources must be specified before annotation changes: %s", s)
		}
	}
	if len(o.resources) < 1 {
		return fmt.Errorf("one or more resources must be specified as <resource> <name> or <resource>/<name>")
	}
	if len(annotationArgs) < 1 {
		return fmt.Errorf("at least one annotation update is required")
	}

	if o.newAnnotations, o.removeAnnotations, err = parseAnnotations(annotationArgs); err != nil {
		return err
	}

	mapper, typer := f.Object()
	o.builder = resource.NewBuilder(mapper, typer, f.ClientMapperForCommand()).
		ContinueOnError().
		NamespaceParam(namespace).DefaultNamespace().
		ResourceTypeOrNameArgs(o.all, o.resources...).
		Flatten().
		Latest()

	return nil
}

// Validate checks to the AnnotateOptions to see if there is sufficient information run the command.
func (o AnnotateOptions) Validate(args []string) error {
	if err := validateAnnotations(o.removeAnnotations, o.newAnnotations); err != nil {
		return err
	}

	// only apply resource version locking on a single resource
	if len(o.resources) > 1 && len(o.resourceVersion) > 0 {
		return fmt.Errorf("--resource-version may only be used with a single resource")
	}

	return nil
}

// RunAnnotate does the work
func (o AnnotateOptions) RunAnnotate() error {
	r := o.builder.Do()
	if err := r.Err(); err != nil {
		return err
	}
	return r.Visit(func(info *resource.Info) error {
		_, err := cmdutil.UpdateObject(info, func(obj runtime.Object) error {
			err := o.updateAnnotations(obj)
			if err != nil {
				return err
			}
			return nil
		})
		if err != nil {
			return err
		}
		return nil
	})
}

// parseAnnotations retrieves new and remove annotations from annotation args
func parseAnnotations(annotationArgs []string) (map[string]string, []string, error) {
	var invalidBuf bytes.Buffer
	newAnnotations := map[string]string{}
	removeAnnotations := []string{}
	for _, annotationArg := range annotationArgs {
		if strings.Index(annotationArg, "=") != -1 {
			parts := strings.SplitN(annotationArg, "=", 2)
			if len(parts) != 2 || len(parts[1]) == 0 {
				if invalidBuf.Len() > 0 {
					invalidBuf.WriteString(", ")
				}
				invalidBuf.WriteString(fmt.Sprintf(annotationArg))
			} else {
				newAnnotations[parts[0]] = parts[1]
			}
		} else if strings.HasSuffix(annotationArg, "-") {
			removeAnnotations = append(removeAnnotations, annotationArg[:len(annotationArg)-1])
		} else {
			if invalidBuf.Len() > 0 {
				invalidBuf.WriteString(", ")
			}
			invalidBuf.WriteString(fmt.Sprintf(annotationArg))
		}
	}
	if invalidBuf.Len() > 0 {
		return newAnnotations, removeAnnotations, fmt.Errorf("invalid annotation format: %s", invalidBuf.String())
	}

	return newAnnotations, removeAnnotations, nil
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
func validateNoAnnotationOverwrites(meta *api.ObjectMeta, annotations map[string]string) error {
	var buf bytes.Buffer
	for key := range annotations {
		if value, found := meta.Annotations[key]; found {
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
	meta, err := api.ObjectMetaFor(obj)
	if err != nil {
		return err
	}
	if !o.overwrite {
		if err := validateNoAnnotationOverwrites(meta, o.newAnnotations); err != nil {
			return err
		}
	}

	if meta.Annotations == nil {
		meta.Annotations = make(map[string]string)
	}

	for key, value := range o.newAnnotations {
		meta.Annotations[key] = value
	}
	for _, annotation := range o.removeAnnotations {
		delete(meta.Annotations, annotation)
	}

	if len(o.resourceVersion) != 0 {
		meta.ResourceVersion = o.resourceVersion
	}
	return nil
}
