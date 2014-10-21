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

	"fmt"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl"
	"github.com/spf13/cobra"
)

func NewCmdNamespace(out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "namespace [<namespace>]",
		Short: "Set and view the current Kubernetes namespace",
		Long: `Set and view the current Kubernetes namespace scope for command line requests.

A Kubernetes namespace subdivides the cluster into groups of logically related pods, services, and replication controllers.

Examples:
  $ kubectl namespace 
  Using namespace default

  $ kubectl namespace other
  Set current namespace to other`,
		Run: func(cmd *cobra.Command, args []string) {
			nsPath := getFlagString(cmd, "ns-path")
			var err error
			var ns *kubectl.NamespaceInfo
			switch len(args) {
			case 0:
				ns, err = kubectl.LoadNamespaceInfo(nsPath)
				fmt.Printf("Using namespace %s\n", ns.Namespace)
			case 1:
				ns = &kubectl.NamespaceInfo{Namespace: args[0]}
				err = kubectl.SaveNamespaceInfo(nsPath, ns)
				fmt.Printf("Set current namespace to %s\n", ns.Namespace)
			default:
				usageError(cmd, "kubectl namespace [<namespace>]")
			}
			checkErr(err)
		},
	}
	return cmd
}
