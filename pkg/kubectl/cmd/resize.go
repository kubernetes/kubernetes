/*
Copyright 2014 Google Inc. All rights reserved.

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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/cmd/util"
	"github.com/spf13/cobra"
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
)

func (f *Factory) NewCmdResize(out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "resize [--resource-version=<version>] [--current-replicas=<count>] --replicas=<count> <resource> <id>",
		Short:   "Set a new size for a Replication Controller.",
		Long:    resize_long,
		Example: resize_example,
		Run: func(cmd *cobra.Command, args []string) {
			count := util.GetFlagInt(cmd, "replicas")
			if len(args) != 2 || count < 0 {
				usageError(cmd, "--replicas=<count> <resource> <id>")
			}

			cmdNamespace, err := f.DefaultNamespace(cmd)
			checkErr(err)

			mapper, _ := f.Object(cmd)
			// TODO: use resource.Builder instead
			mapping, namespace, name := util.ResourceFromArgs(cmd, args, mapper, cmdNamespace)

			resizer, err := f.Resizer(cmd, mapping)
			checkErr(err)

			resourceVersion := util.GetFlagString(cmd, "resource-version")
			currentSize := util.GetFlagInt(cmd, "current-replicas")
			s, err := resizer.Resize(namespace, name, &kubectl.ResizePrecondition{currentSize, resourceVersion}, uint(count))
			checkErr(err)
			fmt.Fprintf(out, "%s\n", s)
		},
	}
	cmd.Flags().String("resource-version", "", "Precondition for resource version. Requires that the current resource version match this value in order to resize.")
	cmd.Flags().Int("current-replicas", -1, "Precondition for current size. Requires that the current size of the replication controller match this value in order to resize.")
	cmd.Flags().Int("replicas", -1, "The new desired number of replicas. Required.")
	return cmd
}
