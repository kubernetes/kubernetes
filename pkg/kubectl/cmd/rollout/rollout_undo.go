/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"io"

	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/runtime"

	"github.com/spf13/cobra"
)

// UndoOptions is the start of the data required to perform the operation.  As new fields are added, add them here instead of
// referencing the cmd.Flags()
type UndoOptions struct {
	Rollbacker kubectl.Rollbacker
	Mapper     meta.RESTMapper
	Typer      runtime.ObjectTyper
	Info       *resource.Info
	ToRevision int64

	Out       io.Writer
	Filenames []string
	Recursive bool
}

const (
	undo_long    = `Rollback to a previous rollout.`
	undo_example = `# Rollback to the previous deployment
kubectl rollout undo deployment/abc

# Rollback to deployment revision 3
kubectl rollout undo deployment/abc --to-revision=3`
)

func NewCmdRolloutUndo(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	options := &UndoOptions{}

	cmd := &cobra.Command{
		Use:     "undo (TYPE NAME | TYPE/NAME) [flags]",
		Short:   "undoes a previous rollout",
		Long:    undo_long,
		Example: undo_example,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.CompleteUndo(f, cmd, out, args))
			cmdutil.CheckErr(options.RunUndo())
		},
	}

	cmd.Flags().Int64("to-revision", 0, "The revision to rollback to. Default to 0 (last revision).")
	usage := "Filename, directory, or URL to a file identifying the resource to get from a server."
	kubectl.AddJsonFilenameFlag(cmd, &options.Filenames, usage)
	cmdutil.AddRecursiveFlag(cmd, &options.Recursive)
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

	infos, err := resource.NewBuilder(o.Mapper, o.Typer, resource.ClientMapperFunc(f.ClientForMapping), f.Decoder(true)).
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, o.Recursive, o.Filenames...).
		ResourceTypeOrNameArgs(true, args...).
		Latest().
		Flatten().
		Do().
		Infos()
	if err != nil {
		return err
	}

	if len(infos) != 1 {
		return fmt.Errorf("rollout undo is only supported on individual resources - %d resources were found", len(infos))
	}
	o.Info = infos[0]
	o.Rollbacker, err = f.Rollbacker(o.Info.ResourceMapping())
	return err
}

func (o *UndoOptions) RunUndo() error {
	result, err := o.Rollbacker.Rollback(o.Info.Namespace, o.Info.Name, nil, o.ToRevision, o.Info.Object)
	if err != nil {
		return err
	}
	cmdutil.PrintSuccess(o.Mapper, false, o.Out, o.Info.Mapping.Resource, o.Info.Name, result)
	return nil
}
