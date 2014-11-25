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

func (f *Factory) NewCmdCreate(out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "create -f filename",
		Short: "Create a resource by filename or stdin",
		Long: `Create a resource by filename or stdin.

JSON and YAML formats are accepted.

Examples:
  $ kubectl create -f pod.json
  <create a pod using the data in pod.json>

  $ cat pod.json | kubectl create -f -
  <create a pod based on the json passed into stdin>`,
		Run: func(cmd *cobra.Command, args []string) {
			filename := GetFlagString(cmd, "filename")
			if len(filename) == 0 {
				usageError(cmd, "Must specify filename to create")
			}
			mapping, namespace, name, data := ResourceFromFile(filename, f.Typer, f.Mapper)
			client, err := f.Client(cmd, mapping)
			checkErr(err)

			// use the default namespace if not specified, or check for conflict with the file's namespace
			if len(namespace) == 0 {
				namespace = getKubeNamespace(cmd)
			} else {
				err = CompareNamespaceFromFile(cmd, namespace)
				checkErr(err)
			}

			err = kubectl.NewRESTHelper(client, mapping).Create(namespace, true, data)
			checkErr(err)
			fmt.Fprintf(out, "%s\n", name)
		},
	}
	cmd.Flags().StringP("filename", "f", "", "Filename or URL to file to use to create the resource")
	return cmd
}
