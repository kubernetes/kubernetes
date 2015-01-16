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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/resource"
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
  <update a pod based on the json passed into stdin>
  
  $ kubectl update pods my-pod --patch='{ "apiVersion": "v1beta1", "desiredState": { "manifest": [{ "cpu": 100 }]}}'
  <update a pod by downloading it, applying the patch, then updating, requires apiVersion be specified>`,
		Run: func(cmd *cobra.Command, args []string) {
			filename := GetFlagString(cmd, "filename")
			patch := GetFlagString(cmd, "patch")
			if len(filename) == 0 && len(patch) == 0 {
				usageError(cmd, "Must specify --filename or --patch to update")
			}
			if len(filename) != 0 && len(patch) != 0 {
				usageError(cmd, "Can not specify both --filename and --patch")
			}
			var name string
			if len(filename) > 0 {
				name = updateWithFile(cmd, f, filename)
			} else {
				name = updateWithPatch(cmd, args, f, patch)
			}
			fmt.Fprintf(out, "%s\n", name)
		},
	}
	cmd.Flags().StringP("filename", "f", "", "Filename or URL to file to use to update the resource")
	cmd.Flags().String("patch", "", "A JSON document to override the existing resource.  The resource is downloaded, then patched with the JSON, the updated")
	return cmd
}

func updateWithPatch(cmd *cobra.Command, args []string, f *Factory, patch string) string {
	mapper, _ := f.Object(cmd)
	mapping, namespace, name := ResourceFromArgs(cmd, args, mapper)
	client, err := f.RESTClient(cmd, mapping)
	checkErr(err)

	helper := resource.NewHelper(client, mapping)
	obj, err := helper.Get(namespace, name)
	checkErr(err)

	Merge(obj, patch, mapping.Kind)

	data, err := helper.Codec.Encode(obj)
	checkErr(err)

	err = helper.Update(namespace, name, true, data)
	checkErr(err)
	return name
}

func updateWithFile(cmd *cobra.Command, f *Factory, filename string) string {
	schema, err := f.Validator(cmd)
	checkErr(err)
	mapper, typer := f.Object(cmd)

	mapping, namespace, name, data := ResourceFromFile(cmd, filename, typer, mapper, schema)

	client, err := f.RESTClient(cmd, mapping)
	checkErr(err)

	err = CompareNamespaceFromFile(cmd, namespace)
	checkErr(err)

	err = resource.NewHelper(client, mapping).Update(namespace, name, true, data)
	checkErr(err)

	return name
}
