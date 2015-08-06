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
	"fmt"
	"io"
	"strings"

	"github.com/spf13/cobra"
	"k8s.io/kubernetes/pkg/api"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
)

const (
	label_long = `Update the labels on a resource.

A label must begin with a letter or number, and may contain letters, numbers, hyphens, dots, and underscores, up to %[1]d characters.
If --overwrite is true, then existing labels can be overwritten, otherwise attempting to overwrite a label will result in an error.
If --resource-version is specified, then updates will use this resource version, otherwise the existing resource-version will be used.`
	label_example = `// Update pod 'foo' with the label 'unhealthy' and the value 'true'.
$ kubectl label pods foo unhealthy=true

// Update pod 'foo' with the label 'status' and the value 'unhealthy', overwriting any existing value.
$ kubectl label --overwrite pods foo status=unhealthy

// Update all pods in the namespace
$ kubectl label pods --all status=unhealthy

// Update pod 'foo' only if the resource is unchanged from version 1.
$ kubectl label pods foo status=unhealthy --resource-version=1

// Update pod 'foo' by removing a label named 'bar' if it exists.
// Does not require the --overwrite flag.
$ kubectl label pods foo bar-`
)

func NewCmdLabel(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "label [--overwrite] TYPE NAME KEY_1=VAL_1 ... KEY_N=VAL_N [--resource-version=version]",
		Short:   "Update the labels on a resource",
		Long:    fmt.Sprintf(label_long, util.LabelValueMaxLength),
		Example: label_example,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunLabel(f, out, cmd, args)
			cmdutil.CheckErr(err)
		},
	}
	cmdutil.AddPrinterFlags(cmd)
	cmd.Flags().Bool("overwrite", false, "If true, allow labels to be overwritten, otherwise reject label updates that overwrite existing labels.")
	cmd.Flags().StringP("selector", "l", "", "Selector (label query) to filter on")
	cmd.Flags().Bool("all", false, "select all resources in the namespace of the specified resource types")
	cmd.Flags().String("resource-version", "", "If non-empty, the labels update will only succeed if this is the current resource-version for the object. Only valid when specifying a single resource.")
	return cmd
}

func validateNoOverwrites(meta *api.ObjectMeta, labels map[string]string) error {
	for key := range labels {
		if value, found := meta.Labels[key]; found {
			return fmt.Errorf("'%s' already has a value (%s), and --overwrite is false", key, value)
		}
	}
	return nil
}

func parseLabels(spec []string) (map[string]string, []string, error) {
	labels := map[string]string{}
	var remove []string
	for _, labelSpec := range spec {
		if strings.Index(labelSpec, "=") != -1 {
			parts := strings.Split(labelSpec, "=")
			if len(parts) != 2 || len(parts[1]) == 0 || !util.IsValidLabelValue(parts[1]) {
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

func RunLabel(f *cmdutil.Factory, out io.Writer, cmd *cobra.Command, args []string) error {
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
	if len(resources) < 1 {
		return cmdutil.UsageError(cmd, "one or more resources must be specified as <resource> <name> or <resource>/<name>")
	}
	if len(labelArgs) < 1 {
		return cmdutil.UsageError(cmd, "at least one label update is required")
	}

	selector := cmdutil.GetFlagString(cmd, "selector")
	all := cmdutil.GetFlagBool(cmd, "all")
	overwrite := cmdutil.GetFlagBool(cmd, "overwrite")
	resourceVersion := cmdutil.GetFlagString(cmd, "resource-version")

	cmdNamespace, _, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	labels, remove, err := parseLabels(labelArgs)
	if err != nil {
		return cmdutil.UsageError(cmd, err.Error())
	}

	mapper, typer := f.Object()
	b := resource.NewBuilder(mapper, typer, f.ClientMapperForCommand()).
		ContinueOnError().
		NamespaceParam(cmdNamespace).DefaultNamespace().
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
	return r.Visit(func(info *resource.Info) error {
		obj, err := cmdutil.UpdateObject(info, func(obj runtime.Object) error {
			err := labelFunc(obj, overwrite, resourceVersion, labels, remove)
			if err != nil {
				return err
			}
			return nil
		})
		if err != nil {
			return err
		}

		printer, err := f.PrinterForMapping(cmd, info.Mapping, false)
		if err != nil {
			return err
		}
		return printer.PrintObj(obj, out)
	})
}
