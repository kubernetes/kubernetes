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

	"github.com/spf13/cobra"
)

// UndoOptions is the start of the data required to perform the operation.  As new fields are added, add them here instead of
// referencing the cmd.Flags()
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
	undo_long = templates.LongDesc(`
		Rollback to a previous rollout.`)

	undo_example = templates.Examples(`
		# Rollback to the previous deployment
		kubectl rollout undo deployment/abc

		# Rollback to deployment revision 3
		kubectl rollout undo deployment/abc --to-revision=3

		# Rollback to the previous deployment with dry-run
		kubectl rollout undo --dry-run=true deployment/abc`)
)

func NewCmdRolloutUndo(f cmdutil.Factory, out io.Writer) *cobra.Command {
	options := &UndoOptions{}

	validArgs := []string{"deployment"}
	argAliases := kubectl.ResourceAliases(validArgs)

	cmd := &cobra.Command{
		Use:     "undo (TYPE NAME | TYPE/NAME) [flags]",
		Short:   "Undo a previous rollout",
		Long:    undo_long,
		Example: undo_example,
		Run: func(cmd *cobra.Command, args []string) {
			allErrs := []error{}
			err := options.CompleteUndo(f, cmd, out, args)
			if err != nil {
				allErrs = append(allErrs, err)
			}
			err = options.RunUndo()
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

func (o *UndoOptions) CompleteUndo(f cmdutil.Factory, cmd *cobra.Command, out io.Writer, args []string) error {
	if len(args) == 0 && cmdutil.IsFilenameEmpty(o.Filenames) {
		return cmdutil.UsageError(cmd, "Required resource not specified.")
	}

	o.ToRevision = cmdutil.GetFlagInt64(cmd, "to-revision")
	o.Mapper, o.Typer = f.Object()
	o.Out = out
	o.DryRun = cmdutil.GetFlagBool(cmd, "dry-run")

	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	r := resource.NewBuilder(o.Mapper, o.Typer, resource.ClientMapperFunc(f.ClientForMapping), f.Decoder(true)).
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

func (o *UndoOptions) RunUndo() error {
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
