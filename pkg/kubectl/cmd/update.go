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
	"github.com/spf13/cobra"
)

func (f *Factory) NewCmdUpdate(out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "update -f filename",
		Short: "Update a resource by filename or stdin",
		Long: `Update a resource by filename or stdin.

JSON and YAML formats are accepted.

Examples:
  $ kubectl update -f pod.json
  <update a pod using the data in pod.json>

  $ cat pod.json | kubectl update -f -
  <update a pod based on the json passed into stdin>`,
		Run: func(cmd *cobra.Command, args []string) {
			filename := GetFlagString(cmd, "filename")
			if len(filename) == 0 {
				usageError(cmd, "Must specify filename to update")
			}
			mapping, namespace, name, data := ResourceFromFile(filename, f.Typer, f.Mapper)
			client, err := f.Client(cmd, mapping)
			checkErr(err)

			err = CompareNamespaceFromFile(cmd, namespace)
			checkErr(err)

			err = kubectl.NewRESTHelper(client, mapping).Update(namespace, name, true, data)
			checkErr(err)
			fmt.Fprintf(out, "%s\n", name)
		},
	}
	cmd.Flags().StringP("filename", "f", "", "Filename or URL to file to use to update the resource")
	return cmd
}
