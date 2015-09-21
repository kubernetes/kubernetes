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
	"io"

	"github.com/spf13/cobra"
)

func (f *Factory) NewCmdLog(out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "log <pod> [<container>]",
		Short: "Print the logs for a container in a pod.",
		Long:  "Print the logs for a container in a pod. If the pod has only one container, the container name is optional.",
		Run: func(cmd *cobra.Command, args []string) {
			if len(args) == 0 {
				usageError(cmd, "<pod> is required for log")
			}

			if len(args) > 2 {
				usageError(cmd, "log <pod> [<container>]")
			}

			namespace := GetKubeNamespace(cmd)
			client, err := f.ClientBuilder.Client()
			checkErr(err)

			podID := args[0]

			pod, err := client.Pods(namespace).Get(podID)
			checkErr(err)

			var container string
			if len(args) == 1 {
				if len(pod.Spec.Containers) != 1 {
					usageError(cmd, "<container> is required for pods with multiple containers")
				}

				// Get logs for the only container in the pod
				container = pod.Spec.Containers[0].Name
			} else {
				container = args[1]
			}

			data, err := client.RESTClient.Get().
				Path("proxy/minions").
				Path(pod.Status.Host).
				Path("containerLogs").
				Path(namespace).
				Path(podID).
				Path(container).
				Do().
				Raw()
			checkErr(err)
			out.Write(data)

		},
	}
	return cmd
}
