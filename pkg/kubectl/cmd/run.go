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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl"
	"github.com/spf13/cobra"
)

func (f *Factory) NewCmdRunContainer(out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "run-container <name> --image=<image> [--replicas=replicas] [--dry-run=<bool>]",
		Short: "Run a particular image on the cluster.",
		Long: `Create and run a particular image, possibly replicated.
Creates a replication controller to manage the created container(s)

Examples:
  $ kubectl run-container nginx --image=dockerfile/nginx
  <starts a single instance of nginx>

  $ kubectl run-container nginx --image=dockerfile/nginx --replicas=5
  <starts a replicated instance of nginx>

  $ kubectl run-container nginx --image=dockerfile/nginx --dry-run
  <just print the corresponding API objects, don't actually send them to the apiserver>`,
		Run: func(cmd *cobra.Command, args []string) {
			if len(args) != 1 {
				usageError(cmd, "<name> is required for run")
			}

			namespace := GetKubeNamespace(cmd)
			client, err := f.Client(cmd)
			checkErr(err)

			generatorName := GetFlagString(cmd, "generator")
			generator, found := kubectl.Generators[generatorName]
			if !found {
				usageError(cmd, fmt.Sprintf("Generator: %s not found.", generator))
			}
			names := generator.ParamNames()
			params := kubectl.MakeParams(cmd, names)
			params["name"] = args[0]

			err = kubectl.ValidateParams(names, params)
			checkErr(err)

			controller, err := generator.Generate(params)
			checkErr(err)

			// TODO: extract this flag to a central location, when such a location exists.
			if !GetFlagBool(cmd, "dry-run") {
				controller, err = client.ReplicationControllers(namespace).Create(controller.(*api.ReplicationController))
				checkErr(err)
			}

			err = PrintObject(cmd, controller, f, out)
			checkErr(err)
		},
	}
	AddPrinterFlags(cmd)
	cmd.Flags().String("generator", "run-container/v1", "The name of the api generator that you want to use.  Default 'run-container-controller-v1'")
	cmd.Flags().String("image", "", "The image for the container you wish to run.")
	cmd.Flags().IntP("replicas", "r", 1, "Number of replicas to create for this container. Default 1")
	cmd.Flags().Bool("dry-run", false, "If true, only print the object that would be sent, don't actually do anything")
	cmd.Flags().StringP("labels", "l", "", "Labels to apply to the pod(s) created by this call to run.")
	return cmd
}
