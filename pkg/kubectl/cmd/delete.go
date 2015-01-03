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

	"github.com/golang/glog"
	"github.com/spf13/cobra"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func (f *Factory) NewCmdDelete(out io.Writer) *cobra.Command {
	flags := &struct {
		Filenames util.StringList
	}{}
	cmd := &cobra.Command{
		Use:   "delete ([-f filename] | (<resource> [(<id> | -l <label>)]",
		Short: "Delete a resource by filename, stdin or resource and id",
		Long: `Delete a resource by filename, stdin, resource and id or by resources and label selector.

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

  $ kubectl delete pods,services -l name=myLabel
  <delete pods and services with label name=myLabel>

  $ kubectl delete pod 1234-56-7890-234234-456456
  <delete a pod with ID 1234-56-7890-234234-456456>`,
		Run: func(cmd *cobra.Command, args []string) {
			mapper, typer := f.Object(cmd)
			r := resource.NewBuilder(mapper, typer, ClientMapperForCommand(cmd, f)).
				ContinueOnError().
				NamespaceParam(GetKubeNamespace(cmd)).DefaultNamespace().
				FilenameParam(flags.Filenames...).
				SelectorParam(GetFlagString(cmd, "selector")).
				ResourceTypeOrNameArgs(args...).
				Flatten().
				Do()

			found := 0
			r.IgnoreErrors(errors.IsNotFound).Visit(func(r *resource.Info) error {
				found++
				if err := resource.NewHelper(r.Client, r.Mapping).Delete(r.Namespace, r.Name); err != nil {
					return err
				}
				fmt.Fprintf(out, "%s\n", r.Name)
				return nil
			})
			if found == 0 {
				glog.V(2).Infof("No resource(s) found")
			}
		},
	}
	cmd.Flags().VarP(&flags.Filenames, "filename", "f", "Filename, directory, or URL to a file containing the resource to delete")
	cmd.Flags().StringP("selector", "l", "", "Selector (label query) to filter on")
	return cmd
}
