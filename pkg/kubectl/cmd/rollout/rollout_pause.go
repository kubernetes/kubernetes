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
	"github.com/spf13/cobra"

	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/runtime"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
)

// PauseConfig is the start of the data required to perform the operation.  As new fields are added, add them here instead of
// referencing the cmd.Flags()
type PauseConfig struct {
	PauseObject func(object runtime.Object) (bool, error)
	Mapper      meta.RESTMapper
	Typer       runtime.ObjectTyper
	Infos       []*resource.Info

	Out       io.Writer
	Filenames []string
	Recursive bool
}

var (
	pause_long = dedent.Dedent(`
		Mark the provided resource as paused

		Paused resources will not be reconciled by a controller.
		Use \"kubectl rollout resume\" to resume a paused resource.
		Currently only deployments support being paused.`)

	pause_example = dedent.Dedent(`
		# Mark the nginx deployment as paused. Any current state of
		# the deployment will continue its function, new updates to the deployment will not
		# have an effect as long as the deployment is paused.
		kubectl rollout pause deployment/nginx`)
)

func NewCmdRolloutPause(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	opts := &PauseConfig{}

	cmd := &cobra.Command{
		Use:     "pause RESOURCE",
		Short:   "Mark the provided resource as paused",
		Long:    pause_long,
		Example: pause_example,
		Run: func(cmd *cobra.Command, args []string) {
			allErrs := []error{}
			err := opts.CompletePause(f, cmd, out, args)
			if err != nil {
				allErrs = append(allErrs, err)
			}
			err = opts.RunPause()
			if err != nil {
				allErrs = append(allErrs, err)
			}
			cmdutil.CheckErr(utilerrors.Flatten(utilerrors.NewAggregate(allErrs)))
		},
	}

	usage := "Filename, directory, or URL to a file identifying the resource to get from a server."
	kubectl.AddJsonFilenameFlag(cmd, &opts.Filenames, usage)
	cmdutil.AddRecursiveFlag(cmd, &opts.Recursive)
	return cmd
}

func (o *PauseConfig) CompletePause(f *cmdutil.Factory, cmd *cobra.Command, out io.Writer, args []string) error {
	if len(args) == 0 && len(o.Filenames) == 0 {
		return cmdutil.UsageError(cmd, cmd.Use)
	}

	o.Mapper, o.Typer = f.Object(false)
	o.PauseObject = f.PauseObject
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

	o.Infos, err = r.Infos()
	if err != nil {
		return err
	}
	return nil
}

func (o PauseConfig) RunPause() error {
	allErrs := []error{}
	for _, info := range o.Infos {
		isAlreadyPaused, err := o.PauseObject(info.Object)
		if err != nil {
			allErrs = append(allErrs, cmdutil.AddSourceToErr("pausing", info.Source, err))
			continue
		}
		if isAlreadyPaused {
			cmdutil.PrintSuccess(o.Mapper, false, o.Out, info.Mapping.Resource, info.Name, "already paused")
			continue
		}
		cmdutil.PrintSuccess(o.Mapper, false, o.Out, info.Mapping.Resource, info.Name, "paused")
	}
	return utilerrors.NewAggregate(allErrs)
}
