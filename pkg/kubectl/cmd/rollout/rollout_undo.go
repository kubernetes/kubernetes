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

	"github.com/renstrom/dedent"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/runtime"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"

	"github.com/spf13/cobra"
)

// UndoOptions is the start of the data required to perform the operation.  As new fields are added, add them here instead of
// referencing the cmd.Flags()
type UndoOptions struct {
	Rollbackers []kubectl.Rollbacker
	Mapper      meta.RESTMapper
	Typer       runtime.ObjectTyper
	Infos       []*resource.Info
	ToRevision  int64

	Out       io.Writer
	Filenames []string
	Recursive bool
}

var (
	undo_long = dedent.Dedent(`
		Rollback to a previous rollout.`)
	undo_example = dedent.Dedent(`
		# Rollback to the previous deployment
		kubectl rollout undo deployment/abc

		# Rollback to deployment revision 3
		kubectl rollout undo deployment/abc --to-revision=3`)
)

func NewCmdRolloutUndo(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	opts := &UndoOptions{}

	validArgs := []string{"deployment"}
	argAliases := kubectl.ResourceAliases(validArgs)

	cmd := &cobra.Command{
		Use:     "undo (TYPE NAME | TYPE/NAME) [flags]",
		Short:   "Undo a previous rollout",
		Long:    undo_long,
		Example: undo_example,
		Run: func(cmd *cobra.Command, args []string) {
			allErrs := []error{}
			err := opts.CompleteUndo(f, cmd, out, args)
			if err != nil {
				allErrs = append(allErrs, err)
			}
			err = opts.RunUndo()
			if err != nil {
				allErrs = append(allErrs, err)
			}
			cmdutil.CheckErr(utilerrors.Flatten(utilerrors.NewAggregate(allErrs)))
		},
		ValidArgs:  validArgs,
		ArgAliases: argAliases,
	}

	cmd.Flags().Int64("to-revision", 0, "The revision to rollback to. Default to 0 (last revision).")
	usage := "Filename, directory, or URL to a file identifying the resource to get from a server."
	kubectl.AddJsonFilenameFlag(cmd, &opts.Filenames, usage)
	cmdutil.AddRecursiveFlag(cmd, &opts.Recursive)
	return cmd
}

func (o *UndoOptions) CompleteUndo(f *cmdutil.Factory, cmd *cobra.Command, out io.Writer, args []string) error {
	if len(args) == 0 && len(o.Filenames) == 0 {
		return cmdutil.UsageError(cmd, "Required resource not specified.")
	}

	o.ToRevision = cmdutil.GetFlagInt64(cmd, "to-revision")
	o.Mapper, o.Typer = f.Object(false)
	o.Out = out

	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	r := resource.NewBuilder(o.Mapper, o.Typer, resource.ClientMapperFunc(f.ClientForMapping), f.Decoder(true)).
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, o.Recursive, o.Filenames...).
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
		result, err := o.Rollbackers[ix].Rollback(info.Object, nil, o.ToRevision)
		if err != nil {
			allErrs = append(allErrs, cmdutil.AddSourceToErr("undoing", info.Source, err))
			continue
		}
		cmdutil.PrintSuccess(o.Mapper, false, o.Out, info.Mapping.Resource, info.Name, false, result)
	}
	return utilerrors.NewAggregate(allErrs)
}
