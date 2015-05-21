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
	"time"

	"github.com/spf13/cobra"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl"
	cmdutil "github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/cmd/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/resource"
)

const (
	resize_long = `Set a new size for a Replication Controller.

Resize also allows users to specify one or more preconditions for the resize action.
If --current-replicas or --resource-version is specified, it is validated before the
resize is attempted, and it is guaranteed that the precondition holds true when the
resize is sent to the server.`
	resize_example = `// Resize replication controller named 'foo' to 3.
$ kubectl resize --replicas=3 replicationcontrollers foo

// If the replication controller named foo's current size is 2, resize foo to 3.
$ kubectl resize --current-replicas=2 --replicas=3 replicationcontrollers foo`

	retryFrequency = 100 * time.Millisecond
	retryTimeout   = 10 * time.Second
)

func NewCmdResize(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "resize [--resource-version=version] [--current-replicas=count] --replicas=COUNT RESOURCE ID",
		Short:   "Set a new size for a Replication Controller.",
		Long:    resize_long,
		Example: resize_example,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunResize(f, out, cmd, args)
			cmdutil.CheckErr(err)
		},
	}
	cmd.Flags().String("resource-version", "", "Precondition for resource version. Requires that the current resource version match this value in order to resize.")
	cmd.Flags().Int("current-replicas", -1, "Precondition for current size. Requires that the current size of the replication controller match this value in order to resize.")
	cmd.Flags().Int("replicas", -1, "The new desired number of replicas. Required.")
	cmd.MarkFlagRequired("replicas")
	return cmd
}

func RunResize(f *cmdutil.Factory, out io.Writer, cmd *cobra.Command, args []string) error {
	count := cmdutil.GetFlagInt(cmd, "replicas")
	if len(args) != 2 || count < 0 {
		return cmdutil.UsageError(cmd, "--replicas=COUNT RESOURCE ID")
	}

	cmdNamespace, err := f.DefaultNamespace()
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
	info := infos[0]

	resizer, err := f.Resizer(mapping)
	if err != nil {
		return err
	}

	resourceVersion := cmdutil.GetFlagString(cmd, "resource-version")
	currentSize := cmdutil.GetFlagInt(cmd, "current-replicas")
	precondition := &kubectl.ResizePrecondition{currentSize, resourceVersion}
	retry := &kubectl.RetryParams{Interval: retryFrequency, Timeout: retryTimeout}

	if err := resizer.Resize(info.Namespace, info.Name, uint(count), precondition, retry, nil); err != nil {
		return err
	}
	fmt.Fprint(out, "resized\n")
	return nil
}
