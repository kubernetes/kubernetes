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

	"github.com/spf13/cobra"

	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/runtime"
)

// ResumeConfig is the start of the data required to perform the operation.  As new fields are added, add them here instead of
// referencing the cmd.Flags()
type ResumeConfig struct {
	ResumeObject func(object runtime.Object) (bool, error)
	Mapper       meta.RESTMapper
	Typer        runtime.ObjectTyper
	Info         *resource.Info

	Out       io.Writer
	Filenames []string
	Recursive bool
}

const (
	resume_long = `Resume a paused resource

Paused resources will not be reconciled by a controller. By resuming a
resource, we allow it to be reconciled again.
Currently only deployments support being resumed.`

	resume_example = `# Resume an already paused deployment
kubectl rollout resume deployment/nginx`
)

func NewCmdRolloutResume(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	opts := &ResumeConfig{}

	cmd := &cobra.Command{
		Use:     "resume RESOURCE",
		Short:   "Resume a paused resource",
		Long:    resume_long,
		Example: resume_example,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(opts.CompleteResume(f, cmd, out, args))
			cmdutil.CheckErr(opts.RunResume())
		},
	}

	usage := "Filename, directory, or URL to a file identifying the resource to get from a server."
	kubectl.AddJsonFilenameFlag(cmd, &opts.Filenames, usage)
	cmdutil.AddRecursiveFlag(cmd, &opts.Recursive)
	return cmd
}

func (o *ResumeConfig) CompleteResume(f *cmdutil.Factory, cmd *cobra.Command, out io.Writer, args []string) error {
	if len(args) == 0 && len(o.Filenames) == 0 {
		return cmdutil.UsageError(cmd, cmd.Use)
	}

	o.Mapper, o.Typer = f.Object(false)
	o.ResumeObject = f.ResumeObject
	o.Out = out

	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	infos, err := resource.NewBuilder(o.Mapper, o.Typer, resource.ClientMapperFunc(f.ClientForMapping), f.Decoder(true)).
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, o.Recursive, o.Filenames...).
		ResourceTypeOrNameArgs(true, args...).
		SingleResourceType().
		Latest().
		Do().Infos()
	if err != nil {
		return err
	}
	if len(infos) != 1 {
		return fmt.Errorf("rollout resume is only supported on individual resources - %d resources were found", len(infos))
	}
	o.Info = infos[0]
	return nil
}

func (o ResumeConfig) RunResume() error {
	isAlreadyResumed, err := o.ResumeObject(o.Info.Object)
	if err != nil {
		return err
	}
	if isAlreadyResumed {
		cmdutil.PrintSuccess(o.Mapper, false, o.Out, o.Info.Mapping.Resource, o.Info.Name, "already resumed")
		return nil
	}
	cmdutil.PrintSuccess(o.Mapper, false, o.Out, o.Info.Mapping.Resource, o.Info.Name, "resumed")
	return nil
}
