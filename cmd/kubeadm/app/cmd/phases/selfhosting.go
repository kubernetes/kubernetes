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
	"github.com/spf13/cobra"

	"k8s.io/kubernetes/cmd/kubeadm/app/phases/selfhosting"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
)

// NewCmdSelfhosting returns the self-hosting Cobra command
func NewCmdSelfhosting() *cobra.Command {
	var kubeConfigFile string
	cmd := &cobra.Command{
		Use:     "selfhosting",
		Aliases: []string{"selfhosted"},
		Short:   "Make a kubeadm cluster self-hosted.",
		Run: func(cmd *cobra.Command, args []string) {
			client, err := kubeconfigutil.ClientSetFromFile(kubeConfigFile)
			kubeadmutil.CheckErr(err)

			err = selfhosting.CreateSelfHostedControlPlane(client)
			kubeadmutil.CheckErr(err)
		},
	}

	cmd.Flags().StringVar(&kubeConfigFile, "kubeconfig", "/etc/kubernetes/admin.conf", "The KubeConfig file to use for talking to the cluster")
	return cmd
}
