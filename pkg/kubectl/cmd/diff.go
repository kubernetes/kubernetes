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
	"errors"
	"io"

	"github.com/spf13/cobra"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/runtime"
)

const (
	diff_example = `# Inspect filesystem changes for the only container in pod nginx
$ kubectl diff nginx

# Inspect filesystem changes for the ruby container in the pod web-1
$ kubectl diff -c ruby web-1`
)

type DiffOptions struct {
	Namespace   string
	ResourceArg string
	Options     runtime.Object

	Mapper       meta.RESTMapper
	Typer        runtime.ObjectTyper
	ClientMapper resource.ClientMapper

	DiffForObject func(object, options runtime.Object) (*client.Request, error)

	Out io.Writer
}

// NewCmdDiff creates a new pod logs command
func NewCmdDiff(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	o := &DiffOptions{}
	cmd := &cobra.Command{
		Use:     "diff POD [-c CONTAINER]",
		Short:   "Inspect the filesystem changes for a container in a pod.",
		Long:    "Inspect the filesystem changes for a container in a pod.  If the pod has only one container, the container name is optional.",
		Example: diff_example,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, out, cmd, args))
			if err := o.Validate(); err != nil {
				cmdutil.CheckErr(cmdutil.UsageError(cmd, err.Error()))
			}
			_, err := o.RunDiff()
			cmdutil.CheckErr(err)
		},
		Aliases: []string{"diff"},
	}
	cmd.Flags().StringP("container", "c", "", "Print the logs of this container")
	return cmd
}

func (o *DiffOptions) Complete(f *cmdutil.Factory, out io.Writer, cmd *cobra.Command, args []string) error {
	//containerName := cmdutil.GetFlagString(cmd, "container")
	switch len(args) {
	case 0:
		return cmdutil.UsageError(cmd, "POD is required for diff")
	case 1:
		o.ResourceArg = args[0]
	case 2:
		if cmd.Flag("container").Changed {
			return cmdutil.UsageError(cmd, "only one of -c, [CONTAINER] arg is allowed")
		}
		o.ResourceArg = args[0]
		//containerName = args[1]
	default:
		return cmdutil.UsageError(cmd, "diff POD [-c CONTAINER]")
	}
	var err error
	o.Namespace, _, err = f.DefaultNamespace()
	if err != nil {
		return err
	}

	// TODO need a pod diff options object
	diffOptions := &api.PodLogOptions{}
	o.Options = diffOptions
	o.Mapper, o.Typer = f.Object()
	o.ClientMapper = f.ClientMapperForCommand()
	o.DiffForObject = f.DiffForObject
	o.Out = out
	return nil
}

func (o DiffOptions) Validate() error {
	if len(o.ResourceArg) == 0 {
		return errors.New("a pod must be specified")
	}
	// TODO: need a diff options
	/*
		logsOptions, ok := o.Options.(*api.PodLogOptions)
		if !ok {
			return errors.New("unexpected diff options object")
		}
		if errs := validation.ValidatePodLogOptions(logsOptions); len(errs) > 0 {
			return errs.ToAggregate()
		}
	*/
	return nil
}

// RunDiff retrieves a pod diff
func (o DiffOptions) RunDiff() (int64, error) {
	infos, err := resource.NewBuilder(o.Mapper, o.Typer, o.ClientMapper).
		NamespaceParam(o.Namespace).DefaultNamespace().
		ResourceNames("pods", o.ResourceArg).
		SingleResourceType().
		Do().Infos()
	if err != nil {
		return 0, err
	}
	if len(infos) != 1 {
		return 0, errors.New("expected a resource")
	}
	info := infos[0]

	req, err := o.DiffForObject(info.Object, o.Options)
	if err != nil {
		return 0, err
	}

	readCloser, err := req.Stream()
	if err != nil {
		return 0, err
	}
	defer readCloser.Close()

	return io.Copy(o.Out, readCloser)
}
