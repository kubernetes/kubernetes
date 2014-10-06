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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl"
	"github.com/spf13/cobra"
)

func NewCmdDelete(out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "delete ([-f filename] | (<resource> <id>))",
		Short: "Delete a resource by filename, stdin or resource and id",
		Long: `Delete a resource by filename, stdin or resource and id.

JSON and YAML formats are accepted.

If both a filename and command line arguments are passed, the command line
arguments are used and the filename is ignored.

Note that the delete command does NOT do resource version checks, so if someone
submits an update to a resource right when you submit a delete, their update
will be lost along with the rest of the resource.

Examples:
  $ kubectl delete -f pod.json
  <delete a pod using the type and id pod.json>

  $ cat pod.json | kubectl delete -f -
  <delete a pod based on the type and id in the json passed into stdin>

  $ kubectl delete pod 1234-56-7890-234234-456456
  <delete a pod with ID 1234-56-7890-234234-456456>`,
		Run: func(cmd *cobra.Command, args []string) {
			// If command line args are passed in, use those preferentially.
			if len(args) > 0 && len(args) != 2 {
				usageError(cmd, "If passing in command line parameters, must be resource and id")
			}

			var data []byte
			var err error

			if len(args) == 2 {
				data, err = kubectl.CreateResource(args[0], args[1])
			} else {
				filename := getFlagString(cmd, "filename")
				if len(filename) > 0 {
					data, err = readConfigData(getFlagString(cmd, "filename"))
				}
			}
			checkErr(err)

			if len(data) == 0 {
				usageError(cmd, "Must specify filename or command line params")
			}

			// TODO Add ability to require a resource-version check for delete.
			err = kubectl.Modify(out, getKubeClient(cmd).RESTClient, kubectl.ModifyDelete, data)
			checkErr(err)
		},
	}
	cmd.Flags().StringP("filename", "f", "", "Filename or URL to file to use to delete the resource")
	return cmd
}
