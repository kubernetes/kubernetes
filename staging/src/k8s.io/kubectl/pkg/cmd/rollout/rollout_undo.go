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

package rollout

import (
	"fmt"

	"github.com/spf13/cobra"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/printers"
	"k8s.io/cli-runtime/pkg/resource"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/polymorphichelpers"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

// UndoOptions is the start of the data required to perform the operation.  As new fields are added, add them here instead of
// referencing the cmd.Flags()
type UndoOptions struct {
	PrintFlags *genericclioptions.PrintFlags
	ToPrinter  func(string) (printers.ResourcePrinter, error)

	Builder          func() *resource.Builder
	ToRevision       int64
	DryRunStrategy   cmdutil.DryRunStrategy
	DryRunVerifier   *resource.DryRunVerifier
	Resources        []string
	Namespace        string
	EnforceNamespace bool
	RESTClientGetter genericclioptions.RESTClientGetter

	resource.FilenameOptions
	genericclioptions.IOStreams
}

var (
	undoLong = templates.LongDesc(i18n.T(`
		Roll back to a previous rollout.`))

	undoExample = templates.Examples(`
		# Roll back to the previous deployment
		kubectl rollout undo deployment/abc

		# Roll back to daemonset revision 3
		kubectl rollout undo daemonset/abc --to-revision=3

		# Roll back to the previous deployment with dry-run
		kubectl rollout undo --dry-run=server deployment/abc`)
)

// NewRolloutUndoOptions returns an initialized UndoOptions instance
func NewRolloutUndoOptions(streams genericclioptions.IOStreams) *UndoOptions {
	return &UndoOptions{
		PrintFlags: genericclioptions.NewPrintFlags("rolled back").WithTypeSetter(scheme.Scheme),
		IOStreams:  streams,
		ToRevision: int64(0),
	}
}

// NewCmdRolloutUndo returns a Command instance for the 'rollout undo' sub command
func NewCmdRolloutUndo(f cmdutil.Factory, streams genericclioptions.IOStreams) *cobra.Command {
	o := NewRolloutUndoOptions(streams)

	validArgs := []string{"deployment", "daemonset", "statefulset"}

	cmd := &cobra.Command{
		Use:                   "undo (TYPE NAME | TYPE/NAME) [flags]",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Undo a previous rollout"),
		Long:                  undoLong,
		Example:               undoExample,
		ValidArgsFunction:     util.SpecifiedResourceTypeAndNameCompletionFunc(f, validArgs),
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.RunUndo())
		},
	}

	cmd.Flags().Int64Var(&o.ToRevision, "to-revision", o.ToRevision, "The revision to rollback to. Default to 0 (last revision).")
	usage := "identifying the resource to get from a server."
	cmdutil.AddFilenameOptionFlags(cmd, &o.FilenameOptions, usage)
	cmdutil.AddDryRunFlag(cmd)
	o.PrintFlags.AddFlags(cmd)
	return cmd
}

// Complete completes all the required options
func (o *UndoOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	o.Resources = args
	var err error
	o.DryRunStrategy, err = cmdutil.GetDryRunStrategy(cmd)
	if err != nil {
		return err
	}
	dynamicClient, err := f.DynamicClient()
	if err != nil {
		return err
	}
	o.DryRunVerifier = resource.NewDryRunVerifier(dynamicClient, f.OpenAPIGetter())

	if o.Namespace, o.EnforceNamespace, err = f.ToRawKubeConfigLoader().Namespace(); err != nil {
		return err
	}

	o.ToPrinter = func(operation string) (printers.ResourcePrinter, error) {
		o.PrintFlags.NamePrintFlags.Operation = operation
		cmdutil.PrintFlagsWithDryRunStrategy(o.PrintFlags, o.DryRunStrategy)
		return o.PrintFlags.ToPrinter()
	}

	o.RESTClientGetter = f
	o.Builder = f.NewBuilder

	return err
}

func (o *UndoOptions) Validate() error {
	if len(o.Resources) == 0 && cmdutil.IsFilenameSliceEmpty(o.Filenames, o.Kustomize) {
		return fmt.Errorf("required resource not specified")
	}
	return nil
}

// RunUndo performs the execution of 'rollout undo' sub command
func (o *UndoOptions) RunUndo() error {
	r := o.Builder().
		WithScheme(scheme.Scheme, scheme.Scheme.PrioritizedVersionsAllGroups()...).
		NamespaceParam(o.Namespace).DefaultNamespace().
		FilenameParam(o.EnforceNamespace, &o.FilenameOptions).
		ResourceTypeOrNameArgs(true, o.Resources...).
		ContinueOnError().
		Latest().
		Flatten().
		Do()
	if err := r.Err(); err != nil {
		return err
	}

	err := r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}
		rollbacker, err := polymorphichelpers.RollbackerFn(o.RESTClientGetter, info.ResourceMapping())
		if err != nil {
			return err
		}

		if o.DryRunStrategy == cmdutil.DryRunServer {
			if err := o.DryRunVerifier.HasSupport(info.Mapping.GroupVersionKind); err != nil {
				return err
			}
		}
		result, err := rollbacker.Rollback(info.Object, nil, o.ToRevision, o.DryRunStrategy)
		if err != nil {
			return err
		}

		printer, err := o.ToPrinter(result)
		if err != nil {
			return err
		}

		return printer.PrintObj(info.Object, o.Out)
	})

	return err
}
