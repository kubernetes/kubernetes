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

	cmdutil "github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/cmd/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/spf13/cobra"
)

func (f *Factory) NewCmdUpdate(out io.Writer) *cobra.Command {
	flags := &struct {
		Filenames util.StringList
	}{}
	cmd := &cobra.Command{
		Use:   "update -f filename",
		Short: "Update a resource by filename or stdin.",
		Long: `Update a resource by filename or stdin.

JSON and YAML formats are accepted.

Examples:

    $ kubectl update -f pod.json
    // Update a pod using the data in pod.json.

    $ cat pod.json | kubectl update -f -
    // Update a pod based on the JSON passed into stdin.

    $ kubectl update pods my-pod --patch='{ "apiVersion": "v1beta1", "desiredState": { "manifest": [{ "cpu": 100 }]}}'
    // Update a pod by downloading it, applying the patch, then updating. Requires apiVersion be specified.`,
		Run: func(cmd *cobra.Command, args []string) {
			schema, err := f.Validator(cmd)
			checkErr(err)

			cmdNamespace, err := f.DefaultNamespace(cmd)
			checkErr(err)

			mapper, typer := f.Object(cmd)
			r := resource.NewBuilder(mapper, typer, f.ClientMapperForCommand(cmd)).
				ContinueOnError().
				NamespaceParam(cmdNamespace).RequireNamespace().
				FilenameParam(flags.Filenames...).
				Flatten().
				Do()
			checkErr(r.Err())

			patch := cmdutil.GetFlagString(cmd, "patch")
			if len(flags.Filenames) == 0 && len(patch) == 0 {
				usageError(cmd, "Must specify --filename or --patch to update")
			}
			if len(flags.Filenames) != 0 && len(patch) != 0 {
				usageError(cmd, "Can not specify both --filename and --patch")
			}

			// TODO: Make patching work with -f, updating with patched JSON input files
			if len(flags.Filenames) == 0 {
				name := updateWithPatch(cmd, args, f, patch)
				fmt.Fprintf(out, "%s\n", name)
				return
			}

			err = r.Visit(func(info *resource.Info) error {
				data, err := info.Mapping.Codec.Encode(info.Object)
				if err != nil {
					return err
				}
				if err := schema.ValidateBytes(data); err != nil {
					return err
				}
				obj, err := resource.NewHelper(info.Client, info.Mapping).Update(info.Namespace, info.Name, true, data)
				if err != nil {
					return err
				}
				info.Refresh(obj, true)
				fmt.Fprintf(out, "%s\n", info.Name)
				return nil
			})
			checkErr(err)

		},
	}
	cmd.Flags().VarP(&flags.Filenames, "filename", "f", "Filename, directory, or URL to file to use to update the resource.")
	cmd.Flags().String("patch", "", "A JSON document to override the existing resource. The resource is downloaded, patched with the JSON, then updated.")
	return cmd
}

func updateWithPatch(cmd *cobra.Command, args []string, f *Factory, patch string) string {
	cmdNamespace, err := f.DefaultNamespace(cmd)
	checkErr(err)

	mapper, _ := f.Object(cmd)
	mapping, namespace, name := cmdutil.ResourceFromArgs(cmd, args, mapper, cmdNamespace)
	client, err := f.RESTClient(cmd, mapping)
	checkErr(err)

	helper := resource.NewHelper(client, mapping)
	obj, err := helper.Get(namespace, name)
	checkErr(err)

	cmdutil.Merge(obj, patch, mapping.Kind)

	data, err := helper.Codec.Encode(obj)
	checkErr(err)

	obj, err = helper.Update(namespace, name, true, data)
	checkErr(err)
	return name
}
