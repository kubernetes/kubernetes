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

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/runtime"
)

// SelectorOptions is the start of the data required to perform the operation.  As new fields are added, add them here instead of
// referencing the cmd.Flags()
type SelectorOptions struct {
	Mapper      meta.RESTMapper
	Typer       runtime.ObjectTyper
	Encoder     runtime.Encoder
	Out         io.Writer
	Filenames   []string
	Recursive   bool
	ShortOutput bool
	All         bool
	Record      bool
	ChangeCause string
	Local       bool
	DryRun      bool
	Resources   []string
	Selector    *unversioned.LabelSelector
	PrintObject func(mapper meta.RESTMapper, obj runtime.Object, out io.Writer) error
}

var (
	selectorLong = dedent.Dedent(`
		Update the selectors on a resource.

		A selector must begin with a letter or number, and may contain letters, numbers, hyphens, dots, and underscores, up to %[1]d characters.
		If --resource-version is specified, then updates will use this resource version, otherwise the existing resource-version will be used.`)
	selectorExample = dedent.Dedent(`
		# Update rc 'foo' with the selector 'unhealthy' and the value 'true'.
		kubectl set selector rc foo unhealthy=true

		# Update all deployments in the namespace
		kubectl set selector deployment --all 'environment in (production, qa)'

		# Update a rs identified by the type and name in "rs.json"
		kubectl set selector -f rs.json 'environment notin (frontend)'

		# Update service 'foo' only if the resource is unchanged from version 1.
		kubectl set selector service foo status=unhealthy --resource-version=1`)
)

// NewCmdSelector is the "set selector" command.
func NewCmdSelector(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	options := &SelectorOptions{
		Out: out,
	}

	cmd := &cobra.Command{
		Use:     "selector (-f FILENAME | TYPE NAME) EXPRESSIONS [--resource-version=version]",
		Short:   "Update the selectors on a resource",
		Long:    fmt.Sprintf(selectorLong),
		Example: selectorExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.Complete(f, cmd, args))
			cmdutil.CheckErr(options.Validate())
			cmdutil.CheckErr(options.Run(f))
		},
	}
	cmdutil.AddPrinterFlags(cmd)
	cmd.Flags().Bool("all", false, "Select all resources in the namespace of the specified resource types")
	cmd.Flags().String("resource-version", "", "If non-empty, the selectors update will only succeed if this is the current resource-version for the object. Only valid when specifying a single resource.")
	usage := "Filename, directory, or URL to a file identifying the resource to update the selectors"
	kubectl.AddJsonFilenameFlag(cmd, &options.Filenames, usage)
	cmdutil.AddRecursiveFlag(cmd, &options.Recursive)
	cmdutil.AddDryRunFlag(cmd)
	cmdutil.AddRecordFlag(cmd)
	cmdutil.AddInclude3rdPartyFlags(cmd)

	return cmd
}

// Complete assigns the SelectorOptions from args.
func (o *SelectorOptions) Complete(f *cmdutil.Factory, cmd *cobra.Command, args []string) error {
	o.Mapper, o.Typer = f.Object(cmdutil.GetIncludeThirdPartyAPIs(cmd))
	o.Encoder = f.JSONEncoder()
	o.ChangeCause = f.Command()
	o.ShortOutput = cmdutil.GetFlagString(cmd, "output") == "name"
	o.Local = cmdutil.GetFlagBool(cmd, "local")
	o.All = cmdutil.GetFlagBool(cmd, "all")
	o.Record = cmdutil.GetRecordFlag(cmd)
	o.DryRun = cmdutil.GetDryRunFlag(cmd)
	o.PrintObject = func(mapper meta.RESTMapper, obj runtime.Object, out io.Writer) error {
		return f.PrintObject(cmd, o.Mapper, obj, o.Out)
	}

	var err error
	o.Resources, o.Selector, err = getResourcesAndSelector(args)
	if err != nil {
		return err
	}

	return nil
}

// Validate basic inputs
func (o *SelectorOptions) Validate() error {
	if len(o.Resources) < 1 && len(o.Filenames) == 0 {
		return fmt.Errorf("one or more resources must be specified as <resource> <name> or <resource>/<name>")
	}
	return nil
}

// Run the command.
func (o *SelectorOptions) Run(f *cmdutil.Factory) error {
	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	builder := resource.NewBuilder(o.Mapper, o.Typer, resource.ClientMapperFunc(f.ClientForMapping), f.Decoder(true)).
		ContinueOnError().
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, o.Recursive, o.Filenames...).
		Flatten()
	if !o.Local {
		builder = builder.
			ResourceTypeOrNameArgs(o.All, o.Resources...).
			Latest()
	}
	r := builder.Do()
	err = r.Err()
	if err != nil {
		return err
	}

	err = r.Visit(func(info *resource.Info, err error) error {
		patch := &Patch{Info: info}
		CalculatePatch(patch, o.Encoder, func(info *resource.Info) (bool, error) {
			err := updateSelectorForObject(info, *o.Selector)
			return true, err
		})
		if patch.Err != nil {
			return patch.Err
		}
		if o.Local || o.DryRun {
			fmt.Fprintln(o.Out, "running in local/dry-run mode...")
			o.PrintObject(o.Mapper, info.Object, o.Out)
			return nil
		}

		patched, err := resource.NewHelper(info.Client, info.Mapping).Patch(info.Namespace, info.Name, api.StrategicMergePatchType, patch.Patch)
		if err != nil {
			return err
		}

		// record this change (for rollout history)
		if o.Record || cmdutil.ContainsChangeCause(info) {
			if err := cmdutil.RecordChangeCause(patched, o.ChangeCause); err == nil {
				if patched, err = resource.NewHelper(info.Client, info.Mapping).Replace(info.Namespace, info.Name, false, patched); err != nil {
					return fmt.Errorf("changes to %s/%s can't be recorded: %v\n", info.Mapping.Resource, info.Name, err)
				}
			}
		}

		info.Refresh(patched, true)
		cmdutil.PrintSuccess(o.Mapper, false, o.Out, info.Mapping.Resource, info.Name, "selector updated")
		return nil
	})

	return nil
}

func updateSelectorForObject(obj runtime.Object, selector unversioned.LabelSelector) error {
	copyOldSelector := func(dst map[string]string) (map[string]string, error) {
		if len(selector.MatchExpressions) > 0 {
			return nil, fmt.Errorf("match expression %v not supported on this object", selector.MatchExpressions)
		}
		dst = make(map[string]string)
		for label, value := range selector.MatchLabels {
			dst[label] = value
		}
		return dst, nil
	}
	copyNewSelector := func(dst *unversioned.LabelSelector) error {
		dst.MatchLabels = make(map[string]string)
		for label, value := range selector.MatchLabels {
			dst.MatchLabels[label] = value
		}

		dst.MatchExpressions = make([]unversioned.LabelSelectorRequirement, len(selector.MatchExpressions))
		copy(dst.MatchExpressions, selector.MatchExpressions)
		return nil
	}
	var err error
	switch t := obj.(type) {
	case *api.ReplicationController:
		t.Spec.Selector, err = copyOldSelector(t.Spec.Selector)
		return err
	case *api.Service:
		t.Spec.Selector, err = copyOldSelector(t.Spec.Selector)
		return err
	case *extensions.Deployment:
		return copyNewSelector(t.Spec.Selector)
	case *extensions.DaemonSet:
		return copyNewSelector(t.Spec.Selector)
	case *extensions.ReplicaSet:
		return copyNewSelector(t.Spec.Selector)
	case *batch.Job:
		return copyNewSelector(t.Spec.Selector)
	case *api.PersistentVolumeClaim:
		return fmt.Errorf("the object does not allow updates to the selector")
	default:
		return fmt.Errorf("the object does not have a selector")
	}
}

// getResourcesAndSelector retrieves resources and the selector expression from the given args (assuming selectors the last arg)
func getResourcesAndSelector(args []string) (resources []string, selector *unversioned.LabelSelector, err error) {
	if len(args) > 1 {
		resources = args[:len(args)-1]
	}
	selector, err = unversioned.ParseToLabelSelector(args[len(args)-1])
	return
}
