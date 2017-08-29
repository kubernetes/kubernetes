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
	"io"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"

	"github.com/spf13/cobra"
)

// UndoOptions holds command line options required to run the command.
type UndoOptions struct {
	resource.FilenameOptions

	Rollbackers []kubectl.Rollbacker
	Mapper      meta.RESTMapper
	Typer       runtime.ObjectTyper
	Infos       []*resource.Info
	ToRevision  int64
	DryRun      bool

	Out io.Writer
}

var (
	undoLong = templates.LongDesc(`
		Rollback to a previous rollout.`)

	undoExample = templates.Examples(`
		# Rollback to the previous deployment
		kubectl rollout undo deployment/abc

		# Rollback to daemonset revision 3
		kubectl rollout undo daemonset/abc --to-revision=3

		# Rollback to the previous deployment with dry-run
		kubectl rollout undo --dry-run=true deployment/abc`)
)

// NewCmdRolloutUndo creates the `undo` subcommand.
func NewCmdRolloutUndo(f cmdutil.Factory, out io.Writer) *cobra.Command {
	options := &UndoOptions{}

	validArgs := []string{"deployment", "daemonset", "statefulset"}
	argAliases := kubectl.ResourceAliases(validArgs)

	cmd := &cobra.Command{
		Use:     "undo (TYPE NAME | TYPE/NAME) [flags]",
		Short:   i18n.T("Undo a previous rollout"),
		Long:    undoLong,
		Example: undoExample,
		Run: func(cmd *cobra.Command, args []string) {
			allErrs := []error{}
			err := options.Complete(f, cmd, out, args)
			if err != nil {
				allErrs = append(allErrs, err)
			}
			err = options.Run()
			if err != nil {
				allErrs = append(allErrs, err)
			}
			cmdutil.CheckErr(utilerrors.Flatten(utilerrors.NewAggregate(allErrs)))
		},
		ValidArgs:  validArgs,
		ArgAliases: argAliases,
	}

	cmd.Flags().Int64("to-revision", 0, "The revision to rollback to. Default to 0 (last revision).")
	usage := "identifying the resource to get from a server."
	cmdutil.AddFilenameOptionFlags(cmd, &options.FilenameOptions, usage)
	cmdutil.AddDryRunFlag(cmd)
	return cmd
}

// Complete completes all the required options.
func (o *UndoOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, out io.Writer, args []string) error {
	if len(args) == 0 && cmdutil.IsFilenameSliceEmpty(o.Filenames) {
		return cmdutil.UsageErrorf(cmd, "Required resource not specified.")
	}

	o.ToRevision = cmdutil.GetFlagInt64(cmd, "to-revision")
	o.Mapper, o.Typer = f.Object()
	o.Out = out
	o.DryRun = cmdutil.GetFlagBool(cmd, "dry-run")

	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	r := f.NewBuilder().
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, &o.FilenameOptions).
		ResourceTypeOrNameArgs(true, args...).
		ContinueOnError().
		Latest().
		Flatten().
		Do()
	err = r.Err()
	if err != nil {
		return err
	}

	err = r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}
		rollbacker, err := f.Rollbacker(info.ResourceMapping())
		if err != nil {
			return err
		}
		o.Infos = append(o.Infos, info)
		o.Rollbackers = append(o.Rollbackers, rollbacker)
		return nil
	})
	return err
}

// Run implements the actual command.
func (o *UndoOptions) Run() error {
	allErrs := []error{}
	for ix, info := range o.Infos {
		result, err := o.Rollbackers[ix].Rollback(info.Object, nil, o.ToRevision, o.DryRun)
		if err != nil {
			allErrs = append(allErrs, cmdutil.AddSourceToErr("undoing", info.Source, err))
			continue
		}
		cmdutil.PrintSuccess(o.Mapper, false, o.Out, info.Mapping.Resource, info.Name, false, result)
	}
	return utilerrors.NewAggregate(allErrs)
}
