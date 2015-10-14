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
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/spf13/cobra"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/errors"
	"k8s.io/kubernetes/pkg/util/strategicpatch"
	"k8s.io/kubernetes/pkg/util/validation"
)

// LabelOptions is the start of the data required to perform the operation.  As new fields are added, add them here instead of
// referencing the cmd.Flags()
type LabelOptions struct {
	Filenames []string
}

const (
	label_long = `Update the labels on a resource.

A label must begin with a letter or number, and may contain letters, numbers, hyphens, dots, and underscores, up to %[1]d characters.
If --overwrite is true, then existing labels can be overwritten, otherwise attempting to overwrite a label will result in an error.
If --resource-version is specified, then updates will use this resource version, otherwise the existing resource-version will be used.`
	label_example = `# Update pod 'foo' with the label 'unhealthy' and the value 'true'.
$ kubectl label pods foo unhealthy=true

# Update pod 'foo' with the label 'status' and the value 'unhealthy', overwriting any existing value.
$ kubectl label --overwrite pods foo status=unhealthy

# Update all pods in the namespace
$ kubectl label pods --all status=unhealthy

# Update a pod identified by the type and name in "pod.json"
$ kubectl label -f pod.json status=unhealthy

# Update pod 'foo' only if the resource is unchanged from version 1.
$ kubectl label pods foo status=unhealthy --resource-version=1

# Update pod 'foo' by removing a label named 'bar' if it exists.
# Does not require the --overwrite flag.
$ kubectl label pods foo bar-`
)

func NewCmdLabel(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	options := &LabelOptions{}

	// retrieve a list of handled resources from printer as valid args
	validArgs := []string{}
	p, err := f.Printer(nil, false, false, false, false, []string{})
	cmdutil.CheckErr(err)
	if p != nil {
		validArgs = p.HandledResources()
	}

	cmd := &cobra.Command{
		Use:     "label [--overwrite] (-f FILENAME | TYPE NAME) KEY_1=VAL_1 ... KEY_N=VAL_N [--resource-version=version]",
		Short:   "Update the labels on a resource",
		Long:    fmt.Sprintf(label_long, validation.LabelValueMaxLength),
		Example: label_example,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunLabel(f, out, cmd, args, options)
			cmdutil.CheckErr(err)
		},
		ValidArgs: validArgs,
	}
	cmdutil.AddPrinterFlags(cmd)
	cmd.Flags().Bool("overwrite", false, "If true, allow labels to be overwritten, otherwise reject label updates that overwrite existing labels.")
	cmd.Flags().StringP("selector", "l", "", "Selector (label query) to filter on")
	cmd.Flags().Bool("all", false, "select all resources in the namespace of the specified resource types")
	cmd.Flags().String("resource-version", "", "If non-empty, the labels update will only succeed if this is the current resource-version for the object. Only valid when specifying a single resource.")
	usage := "Filename, directory, or URL to a file identifying the resource to update the labels"
	kubectl.AddJsonFilenameFlag(cmd, &options.Filenames, usage)
	cmd.Flags().Bool("dry-run", false, "If true, only print the object that would be sent, without sending it.")

	return cmd
}

func validateNoOverwrites(meta *api.ObjectMeta, labels map[string]string) error {
	allErrs := []error{}
	for key := range labels {
		if value, found := meta.Labels[key]; found {
			allErrs = append(allErrs, fmt.Errorf("'%s' already has a value (%s), and --overwrite is false", key, value))
		}
	}
	return errors.NewAggregate(allErrs)
}

func parseLabels(spec []string) (map[string]string, []string, error) {
	labels := map[string]string{}
	var remove []string
	for _, labelSpec := range spec {
		if strings.Index(labelSpec, "=") != -1 {
			parts := strings.Split(labelSpec, "=")
			if len(parts) != 2 || len(parts[1]) == 0 || !validation.IsValidLabelValue(parts[1]) {
				return nil, nil, fmt.Errorf("invalid label spec: %v", labelSpec)
			}
			labels[parts[0]] = parts[1]
		} else if strings.HasSuffix(labelSpec, "-") {
			remove = append(remove, labelSpec[:len(labelSpec)-1])
		} else {
			return nil, nil, fmt.Errorf("unknown label spec: %v", labelSpec)
		}
	}
	for _, removeLabel := range remove {
		if _, found := labels[removeLabel]; found {
			return nil, nil, fmt.Errorf("can not both modify and remove a label in the same command")
		}
	}
	return labels, remove, nil
}

func labelFunc(obj runtime.Object, overwrite bool, resourceVersion string, labels map[string]string, remove []string) error {
	meta, err := api.ObjectMetaFor(obj)
	if err != nil {
		return err
	}
	if !overwrite {
		if err := validateNoOverwrites(meta, labels); err != nil {
			return err
		}
	}

	if meta.Labels == nil {
		meta.Labels = make(map[string]string)
	}

	for key, value := range labels {
		meta.Labels[key] = value
	}
	for _, label := range remove {
		delete(meta.Labels, label)
	}

	if len(resourceVersion) != 0 {
		meta.ResourceVersion = resourceVersion
	}
	return nil
}

func RunLabel(f *cmdutil.Factory, out io.Writer, cmd *cobra.Command, args []string, options *LabelOptions) error {
	resources, labelArgs := []string{}, []string{}
	first := true
	for _, s := range args {
		isLabel := strings.Contains(s, "=") || strings.HasSuffix(s, "-")
		switch {
		case first && isLabel:
			first = false
			fallthrough
		case !first && isLabel:
			labelArgs = append(labelArgs, s)
		case first && !isLabel:
			resources = append(resources, s)
		case !first && !isLabel:
			return cmdutil.UsageError(cmd, "all resources must be specified before label changes: %s", s)
		}
	}
	if len(resources) < 1 && len(options.Filenames) == 0 {
		return cmdutil.UsageError(cmd, "one or more resources must be specified as <resource> <name> or <resource>/<name>")
	}
	if len(labelArgs) < 1 {
		return cmdutil.UsageError(cmd, "at least one label update is required")
	}

	selector := cmdutil.GetFlagString(cmd, "selector")
	all := cmdutil.GetFlagBool(cmd, "all")
	overwrite := cmdutil.GetFlagBool(cmd, "overwrite")
	resourceVersion := cmdutil.GetFlagString(cmd, "resource-version")

	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	lbls, remove, err := parseLabels(labelArgs)
	if err != nil {
		return cmdutil.UsageError(cmd, err.Error())
	}
	mapper, typer := f.Object()
	b := resource.NewBuilder(mapper, typer, f.ClientMapperForCommand()).
		ContinueOnError().
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, options.Filenames...).
		SelectorParam(selector).
		ResourceTypeOrNameArgs(all, resources...).
		Flatten().
		Latest()

	one := false
	r := b.Do().IntoSingular(&one)
	if err := r.Err(); err != nil {
		return err
	}

	// only apply resource version locking on a single resource
	if !one && len(resourceVersion) > 0 {
		return cmdutil.UsageError(cmd, "--resource-version may only be used with a single resource")
	}

	// TODO: support bulk generic output a la Get
	return r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}

		var outputObj runtime.Object
		if cmdutil.GetFlagBool(cmd, "dry-run") {
			err = labelFunc(info.Object, overwrite, resourceVersion, lbls, remove)
			if err != nil {
				return err
			}
			outputObj = info.Object
		} else {
			name, namespace, obj := info.Name, info.Namespace, info.Object
			oldData, err := json.Marshal(obj)
			if err != nil {
				return err
			}
			if err := labelFunc(obj, overwrite, resourceVersion, lbls, remove); err != nil {
				return err
			}
			newData, err := json.Marshal(obj)
			if err != nil {
				return err
			}
			patchBytes, err := strategicpatch.CreateTwoWayMergePatch(oldData, newData, obj)
			if err != nil {
				return err
			}

			mapping := info.ResourceMapping()
			client, err := f.RESTClient(mapping)
			if err != nil {
				return err
			}
			helper := resource.NewHelper(client, mapping)

			outputObj, err = helper.Patch(namespace, name, api.StrategicMergePatchType, patchBytes)
			if err != nil {
				return err
			}
		}
		outputFormat := cmdutil.GetFlagString(cmd, "output")
		if outputFormat != "" {
			return f.PrintObject(cmd, outputObj, out)
		}
		cmdutil.PrintSuccess(mapper, false, out, info.Mapping.Resource, info.Name, "labeled")
		return nil
	})
}
