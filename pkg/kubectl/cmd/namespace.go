/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

	"github.com/spf13/cobra"
	"k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

// TODO remove once people have been given enough time to notice
func NewCmdNamespace(out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "namespace [namespace]",
		Short: "SUPERSEDED: Set and view the current Kubernetes namespace",
		Long: `SUPERSEDED:  Set and view the current Kubernetes namespace scope for command line requests.

namespace has been superseded by the context.namespace field of .kubeconfig files.  See 'kubectl config set-context --help' for more details.
`,
		Run: func(cmd *cobra.Command, args []string) {
			util.CheckErr(fmt.Errorf("namespace has been superseded by the context.namespace field of .kubeconfig files.  See 'kubectl config set-context --help' for more details."))
		},
	}
	return cmd
}
