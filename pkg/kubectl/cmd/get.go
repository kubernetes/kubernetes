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

func NewCmdGet(out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "get [(-o|--output=)table|json|yaml|template] [-t <file>|--template=<file>] <resource> [<id>]",
		Short: "Display one or many resources",
		Long: `Display one or many resources.

Possible resources include pods (po), replication controllers (rc), services
(se) or minions (mi).

If you specify a Go template, you can use any field defined in pkg/api/types.go.

Examples:
  $ kubectl get pods
  <list all pods in ps output format>

  $ kubectl get replicationController 1234-56-7890-234234-456456
  <list single replication controller in ps output format>

  $ kubectl get -f json pod 1234-56-7890-234234-456456
  <list single pod in json output format>`,
		Run: func(cmd *cobra.Command, args []string) {
			var resource, id string
			if len(args) == 0 {
				usageError(cmd, "Need to supply a resource.")
			}
			if len(args) >= 1 {
				resource = args[0]
			}
			if len(args) >= 2 {
				id = args[1]
			}
			outputFormat := getFlagString(cmd, "output")
			templateFile := getFlagString(cmd, "template")
			selector := getFlagString(cmd, "selector")
			err := kubectl.Get(out, getKubeClient(cmd).RESTClient, getKubeNamespace(cmd), resource, id, selector, outputFormat, getFlagBool(cmd, "no-headers"), templateFile)
			checkErr(err)
		},
	}
	// TODO Add an --output-version lock which can ensure that regardless of the
	// server version, the client output stays the same.
	cmd.Flags().StringP("output", "o", "console", "Output format: console|json|yaml|template")
	cmd.Flags().Bool("no-headers", false, "When output format is console, don't print headers")
	cmd.Flags().StringP("template", "t", "", "Path to template file to use when --output=template")
	cmd.Flags().StringP("selector", "l", "", "Selector (label query) to filter on")
	return cmd
}
