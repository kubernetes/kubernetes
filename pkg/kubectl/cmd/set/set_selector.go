/*
Copyright 2016 The Kubernetes Authors.

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

package set

import (
	"fmt"
	"io"

	"k8s.io/kubernetes/pkg/printers"

	"github.com/golang/glog"
	"github.com/spf13/cobra"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/genericclioptions"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
)

// SelectorOptions is the start of the data required to perform the operation.  As new fields are added, add them here instead of
// referencing the cmd.Flags()
type SetSelectorOptions struct {
	fileOptions resource.FilenameOptions

	PrintFlags  *printers.PrintFlags
	RecordFlags *genericclioptions.RecordFlags

	local  bool
	dryrun bool
	all    bool
	output string

	resources []string
	selector  *metav1.LabelSelector

	out              io.Writer
	ClientForMapping func(mapping *meta.RESTMapping) (resource.RESTClient, error)

	PrintObj printers.ResourcePrinterFunc
	Recorder genericclioptions.Recorder

	builder *resource.Builder
	mapper  meta.RESTMapper
}

var (
	selectorLong = templates.LongDesc(`
		Set the selector on a resource. Note that the new selector will overwrite the old selector if the resource had one prior to the invocation
		of 'set selector'.

		A selector must begin with a letter or number, and may contain letters, numbers, hyphens, dots, and underscores, up to %[1]d characters.
		If --resource-version is specified, then updates will use this resource version, otherwise the existing resource-version will be used.
        Note: currently selectors can only be set on Service objects.`)
	selectorExample = templates.Examples(`
        # set the labels and selector before creating a deployment/service pair.
        kubectl create service clusterip my-svc --clusterip="None" -o yaml --dry-run | kubectl set selector --local -f - 'environment=qa' -o yaml | kubectl create -f -
        kubectl create deployment my-dep -o yaml --dry-run | kubectl label --local -f - environment=qa -o yaml | kubectl create -f -`)
)

func NewSelectorOptions(out io.Writer) *SetSelectorOptions {
	return &SetSelectorOptions{
		PrintFlags:  printers.NewPrintFlags("selector updated"),
		RecordFlags: genericclioptions.NewRecordFlags(),

		Recorder: genericclioptions.NoopRecorder{},

		out: out,
	}
}

// NewCmdSelector is the "set selector" command.
func NewCmdSelector(f cmdutil.Factory, out io.Writer) *cobra.Command {
	o := NewSelectorOptions(out)

	cmd := &cobra.Command{
		Use: "selector (-f FILENAME | TYPE NAME) EXPRESSIONS [--resource-version=version]",
		DisableFlagsInUseLine: true,
		Short:   i18n.T("Set the selector on a resource"),
		Long:    fmt.Sprintf(selectorLong, validation.LabelValueMaxLength),
		Example: selectorExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.RunSelector())
		},
	}

	o.PrintFlags.AddFlags(cmd)
	o.RecordFlags.AddFlags(cmd)

	cmd.Flags().Bool("all", false, "Select all resources, including uninitialized ones, in the namespace of the specified resource types")
	cmd.Flags().Bool("local", false, "If true, set selector will NOT contact api-server but run locally.")
	cmd.Flags().String("resource-version", "", "If non-empty, the selectors update will only succeed if this is the current resource-version for the object. Only valid when specifying a single resource.")
	usage := "the resource to update the selectors"
	cmdutil.AddFilenameOptionFlags(cmd, &o.fileOptions, usage)
	cmdutil.AddDryRunFlag(cmd)
	cmdutil.AddIncludeUninitializedFlag(cmd)

	return cmd
}

// Complete assigns the SelectorOptions from args.
func (o *SetSelectorOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	var err error

	o.RecordFlags.Complete(f.Command(cmd, false))
	o.Recorder, err = o.RecordFlags.ToRecorder()
	if err != nil {
		return err
	}

	o.local = cmdutil.GetFlagBool(cmd, "local")
	o.all = cmdutil.GetFlagBool(cmd, "all")
	o.dryrun = cmdutil.GetDryRunFlag(cmd)
	o.output = cmdutil.GetFlagString(cmd, "output")

	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	mapper, _ := f.Object()
	o.mapper = mapper

	o.resources, o.selector, err = getResourcesAndSelector(args)
	if err != nil {
		return err
	}

	includeUninitialized := cmdutil.ShouldIncludeUninitialized(cmd, false)
	o.builder = f.NewBuilder().
		Internal().
		LocalParam(o.local).
		ContinueOnError().
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, &o.fileOptions).
		IncludeUninitialized(includeUninitialized).
		Flatten()

	if !o.local {
		o.builder.
			ResourceTypeOrNameArgs(o.all, o.resources...).
			Latest()
	} else {
		// if a --local flag was provided, and a resource was specified in the form
		// <resource>/<name>, fail immediately as --local cannot query the api server
		// for the specified resource.
		if len(o.resources) > 0 {
			return resource.LocalResourceError
		}
	}

	if o.dryrun {
		o.PrintFlags.Complete("%s (dry run)")
	}
	printer, err := o.PrintFlags.ToPrinter()
	if err != nil {
		return err
	}
	o.PrintObj = printer.PrintObj

	o.ClientForMapping = func(mapping *meta.RESTMapping) (resource.RESTClient, error) {
		return f.ClientForMapping(mapping)
	}
	return err
}

// Validate basic inputs
func (o *SetSelectorOptions) Validate() error {
	if len(o.resources) < 1 && cmdutil.IsFilenameSliceEmpty(o.fileOptions.Filenames) {
		return fmt.Errorf("one or more resources must be specified as <resource> <name> or <resource>/<name>")
	}
	if o.selector == nil {
		return fmt.Errorf("one selector is required")
	}
	return nil
}

// RunSelector executes the command.
func (o *SetSelectorOptions) RunSelector() error {
	r := o.builder.Do()
	err := r.Err()
	if err != nil {
		return err
	}

	return r.Visit(func(info *resource.Info, err error) error {
		patch := &Patch{Info: info}
		CalculatePatch(patch, cmdutil.InternalVersionJSONEncoder(), func(info *resource.Info) ([]byte, error) {
			versioned := info.AsVersioned()
			patch.Info.Object = versioned
			selectErr := updateSelectorForObject(info.Object, *o.selector)
			if selectErr != nil {
				return nil, selectErr
			}

			// record this change (for rollout history)
			if err := o.Recorder.Record(patch.Info.Object); err != nil {
				glog.V(4).Infof("error recording current command: %v", err)
			}

			return runtime.Encode(cmdutil.InternalVersionJSONEncoder(), info.Object)
		})

		if patch.Err != nil {
			return patch.Err
		}
		if o.local || o.dryrun {
			return o.PrintObj(info.Object, o.out)
		}

		patched, err := resource.NewHelper(info.Client, info.Mapping).Patch(info.Namespace, info.Name, types.StrategicMergePatchType, patch.Patch)
		if err != nil {
			return err
		}

		info.Refresh(patched, true)
		return o.PrintObj(patch.Info.AsVersioned(), o.out)
	})
}

func updateSelectorForObject(obj runtime.Object, selector metav1.LabelSelector) error {
	copyOldSelector := func() (map[string]string, error) {
		if len(selector.MatchExpressions) > 0 {
			return nil, fmt.Errorf("match expression %v not supported on this object", selector.MatchExpressions)
		}
		dst := make(map[string]string)
		for label, value := range selector.MatchLabels {
			dst[label] = value
		}
		return dst, nil
	}
	var err error
	switch t := obj.(type) {
	case *v1.Service:
		t.Spec.Selector, err = copyOldSelector()
	default:
		err = fmt.Errorf("setting a selector is only supported for Services")
	}
	return err
}

// getResourcesAndSelector retrieves resources and the selector expression from the given args (assuming selectors the last arg)
func getResourcesAndSelector(args []string) (resources []string, selector *metav1.LabelSelector, err error) {
	if len(args) == 0 {
		return []string{}, nil, nil
	}
	resources = args[:len(args)-1]
	selector, err = metav1.ParseToLabelSelector(args[len(args)-1])
	return resources, selector, err
}
