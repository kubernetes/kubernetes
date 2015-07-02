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
	"fmt"
	"io"
	"os"

	"github.com/spf13/cobra"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl"
	cmdutil "github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/cmd/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/resource"
)

const (
	scale_long = `Set a new size for a Replication Controller.

Scale also allows users to specify one or more preconditions for the scale action.
If --current-replicas or --resource-version is specified, it is validated before the
scale is attempted, and it is guaranteed that the precondition holds true when the
scale is sent to the server.`
	scale_example = `// Scale replication controller named 'foo' to 3.
$ kubectl scale --replicas=3 replicationcontrollers foo

// If the replication controller named foo's current size is 2, scale foo to 3.
$ kubectl scale --current-replicas=2 --replicas=3 replicationcontrollers foo`
)

// NewCmdScale returns a cobra command with the appropriate configuration and flags to run scale
func NewCmdScale(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use: "scale [--resource-version=version] [--current-replicas=count] --replicas=COUNT RESOURCE NAME",
		// resize is deprecated
		Aliases: []string{"resize"},
		Short:   "Set a new size for a Replication Controller.",
		Long:    scale_long,
		Example: scale_example,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunScale(f, out, cmd, args)
			cmdutil.CheckErr(err)
		},
	}
	cmd.Flags().String("resource-version", "", "Precondition for resource version. Requires that the current resource version match this value in order to scale.")
	cmd.Flags().Int("current-replicas", -1, "Precondition for current size. Requires that the current size of the replication controller match this value in order to scale.")
	cmd.Flags().Int("replicas", -1, "The new desired number of replicas. Required.")
	cmd.MarkFlagRequired("replicas")
	return cmd
}

// RunScale executes the scaling
func RunScale(f *cmdutil.Factory, out io.Writer, cmd *cobra.Command, args []string) error {
	if os.Args[1] == "resize" {
		printDeprecationWarning("scale", "resize")
	}

	count := cmdutil.GetFlagInt(cmd, "replicas")
	if count < 0 {
		return cmdutil.UsageError(cmd, "--replicas=COUNT RESOURCE NAME")
	}

	cmdNamespace, _, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	mapper, typer := f.Object()
	r := resource.NewBuilder(mapper, typer, f.ClientMapperForCommand()).
		ContinueOnError().
		NamespaceParam(cmdNamespace).DefaultNamespace().
		ResourceTypeOrNameArgs(false, args...).
		Flatten().
		Do()
	err = r.Err()
	if err != nil {
		return err
	}
	mapping, err := r.ResourceMapping()
	if err != nil {
		return err
	}

	infos, err := r.Infos()
	if err != nil {
		return err
	}
	if len(infos) > 1 {
		return fmt.Errorf("multiple resources provided: %v", args)
	}
	info := infos[0]

	scaler, err := f.Scaler(mapping)
	if err != nil {
		return err
	}

	resourceVersion := cmdutil.GetFlagString(cmd, "resource-version")
	currentSize := cmdutil.GetFlagInt(cmd, "current-replicas")
	precondition := &kubectl.ScalePrecondition{currentSize, resourceVersion}
	retry := kubectl.NewRetryParams(kubectl.Interval, kubectl.Timeout)
	waitForReplicas := kubectl.NewRetryParams(kubectl.Interval, kubectl.Timeout)
	if err := scaler.Scale(info.Namespace, info.Name, uint(count), precondition, retry, waitForReplicas); err != nil {
		return err
	}
	fmt.Fprint(out, "scaled\n")
	return nil
}
