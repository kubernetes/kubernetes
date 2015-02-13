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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/cmd/util"
	"github.com/spf13/cobra"
)

func (f *Factory) NewCmdRunContainer(out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "run-container <name> --image=<image> [--port=<port>] [--replicas=replicas] [--dry-run=<bool>] [--overrides=<inline-json>]",
		Short: "Run a particular image on the cluster.",
		Long: `Create and run a particular image, possibly replicated.
Creates a replication controller to manage the created container(s).

Examples:

    // Starts a single instance of nginx.
    $ kubectl run-container nginx --image=dockerfile/nginx

    // Starts a replicated instance of nginx.
    $ kubectl run-container nginx --image=dockerfile/nginx --replicas=5

    // Dry run. Print the corresponding API objects without creating them.
    $ kubectl run-container nginx --image=dockerfile/nginx --dry-run
  
    // Start a single instance of nginx, but overload the desired state with a partial set of values parsed from JSON.
    $ kubectl run-container nginx --image=dockerfile/nginx --overrides='{ "apiVersion": "v1beta1", "desiredState": { ... } }'`,
		Run: func(cmd *cobra.Command, args []string) {
			if len(args) != 1 {
				usageError(cmd, "<name> is required for run-container")
			}

			namespace, err := f.DefaultNamespace(cmd)
			checkErr(err)

			client, err := f.Client(cmd)
			checkErr(err)

			generatorName := util.GetFlagString(cmd, "generator")
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

			inline := util.GetFlagString(cmd, "overrides")
			if len(inline) > 0 {
				util.Merge(controller, inline, "ReplicationController")
			}

			// TODO: extract this flag to a central location, when such a location exists.
			if !util.GetFlagBool(cmd, "dry-run") {
				controller, err = client.ReplicationControllers(namespace).Create(controller.(*api.ReplicationController))
				checkErr(err)
			}

			err = f.PrintObject(cmd, controller, out)
			checkErr(err)
		},
	}
	util.AddPrinterFlags(cmd)
	cmd.Flags().String("generator", "run-container/v1", "The name of the API generator to use.  Default is 'run-container-controller/v1'.")
	cmd.Flags().String("image", "", "The image for the container to run.")
	cmd.Flags().IntP("replicas", "r", 1, "Number of replicas to create for this container. Default is 1.")
	cmd.Flags().Bool("dry-run", false, "If true, only print the object that would be sent, without sending it.")
	cmd.Flags().String("overrides", "", "An inline JSON override for the generated object. If this is non-empty, it is used to override the generated object. Requires that the object supply a valid apiVersion field.")
	cmd.Flags().Int("port", -1, "The port that this container exposes.")
	cmd.Flags().StringP("labels", "l", "", "Labels to apply to the pod(s) created by this call to run-container.")
	return cmd
}
