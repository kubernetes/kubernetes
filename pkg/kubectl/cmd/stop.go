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
	"io"

	"github.com/spf13/cobra"
	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
)

const (
	stop_long = `Deprecated: Gracefully shut down a resource by name or filename.

The stop command is deprecated, all its functionalities are covered by delete command.
See 'kubectl delete --help' for more details.

Attempts to shut down and delete a resource that supports graceful termination.
If the resource is scalable it will be scaled to 0 before deletion.`
	stop_example = `// Shut down foo.
$ kubectl stop replicationcontroller foo

// Stop pods and services with label name=myLabel.
$ kubectl stop pods,services -l name=myLabel

// Shut down the service defined in service.json
$ kubectl stop -f service.json

// Shut down all resources in the path/to/resources directory
$ kubectl stop -f path/to/resources`
)

func NewCmdStop(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "stop (-f FILENAME | TYPE (NAME | -l label | --all))",
		Short:   "Deprecated: Gracefully shut down a resource by name or filename.",
		Long:    stop_long,
		Example: stop_example,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(cmdutil.ValidateOutputArgs(cmd))
			cmdutil.CheckErr(RunStop(f, cmd, args, out))
		},
	}
	usage := "Filename, directory, or URL to file of resource(s) to be stopped."
	kubectl.AddJsonFilenameFlag(cmd, usage)
	cmd.Flags().StringP("selector", "l", "", "Selector (label query) to filter on.")
	cmd.Flags().Bool("all", false, "[-all] to select all the specified resources.")
	cmd.Flags().Bool("ignore-not-found", false, "Treat \"resource not found\" as a successful stop.")
	cmd.Flags().Int("grace-period", -1, "Period of time in seconds given to the resource to terminate gracefully. Ignored if negative.")
	cmd.Flags().Duration("timeout", 0, "The length of time to wait before giving up on a delete, zero means determine a timeout from the size of the object")
	cmdutil.AddOutputFlagsForMutation(cmd)
	return cmd
}

func RunStop(f *cmdutil.Factory, cmd *cobra.Command, args []string, out io.Writer) error {
	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	filenames := cmdutil.GetFlagStringSlice(cmd, "filename")
	mapper, typer := f.Object()
	r := resource.NewBuilder(mapper, typer, f.ClientMapperForCommand()).
		ContinueOnError().
		NamespaceParam(cmdNamespace).DefaultNamespace().
		ResourceTypeOrNameArgs(false, args...).
		FilenameParam(enforceNamespace, filenames...).
		SelectorParam(cmdutil.GetFlagString(cmd, "selector")).
		SelectAllParam(cmdutil.GetFlagBool(cmd, "all")).
		Flatten().
		Do()
	if r.Err() != nil {
		return r.Err()
	}
	shortOutput := cmdutil.GetFlagString(cmd, "output") == "name"
	return ReapResult(r, f, out, false, cmdutil.GetFlagBool(cmd, "ignore-not-found"), cmdutil.GetFlagDuration(cmd, "timeout"), cmdutil.GetFlagInt(cmd, "grace-period"), shortOutput, mapper)
}
