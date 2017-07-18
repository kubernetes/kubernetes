/*
Copyright 2017 The Kubernetes Authors.

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

package phases

import (
	"fmt"

	"github.com/spf13/cobra"

	markmasterphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/markmaster"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
)

// NewCmdMarkMaster returns the Cobra command for running the mark-master phase
func NewCmdMarkMaster() *cobra.Command {
	var kubeConfigFile string
	cmd := &cobra.Command{
		Use:     "mark-master <node-name>",
		Short:   "Create KubeConfig files from given credentials.",
		Aliases: []string{"markmaster"},
		RunE: func(_ *cobra.Command, args []string) error {
			if len(args) < 1 || len(args[0]) == 0 {
				return fmt.Errorf("missing required argument node-name")
			}
			if len(args) > 1 {
				return fmt.Errorf("too many arguments, only one argument supported: node-name")
			}
			client, err := kubeconfigutil.ClientSetFromFile(kubeConfigFile)
			if err != nil {
				return err
			}

			nodeName := args[0]
			fmt.Printf("[markmaster] Will mark node %s as master by adding a label and a taint\n", nodeName)

			return markmasterphase.MarkMaster(client, nodeName)
		},
	}

	cmd.Flags().StringVar(&kubeConfigFile, "kubeconfig", "/etc/kubernetes/admin.conf", "The KubeConfig file to use for talking to the cluster")
	return cmd
}
