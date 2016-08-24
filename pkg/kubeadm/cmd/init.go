/*
Copyright 2016 The Kubernetes Authors.

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

package kubecmd

import (
	"fmt"
	"io"

	"github.com/spf13/cobra"

	kubeadmapi "k8s.io/kubernetes/pkg/kubeadm/api"
	kubemaster "k8s.io/kubernetes/pkg/kubeadm/master"
	"k8s.io/kubernetes/pkg/kubeadm/tlsutil"
	kubeadmutil "k8s.io/kubernetes/pkg/kubeadm/util"
)

func NewCmdInit(out io.Writer, params *kubeadmapi.BootstrapParams) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "init",
		Short: "Run this on the first server you deploy onto.",
		RunE: func(cmd *cobra.Command, args []string) error {
			if params.Discovery.ListenIP == "" {
				ip, err := kubeadmutil.GetDefaultHostIP()
				if err != nil {
					return err
				}
				params.Discovery.ListenIP = ip
			}
			if err := kubemaster.CreateTokenAuthFile(params); err != nil {
				return err
			}
			if err := kubemaster.WriteStaticPodManifests(params); err != nil {
				return err
			}
			caKey, caCert, err := kubemaster.CreatePKIAssets(params)
			if err != nil {
				return err
			}
			kubeconfigs, err := kubemaster.CreateCertsAndConfigForClients(params, []string{"kubelet", "admin"}, caKey, caCert)
			if err != nil {
				return err
			}
			for name, kubeconfig := range kubeconfigs {
				if err := kubeadmutil.WriteKubeconfigIfNotExists(params, name, kubeconfig); err != nil {
					out.Write([]byte(fmt.Sprintf("Unable to write admin for master:\n%s\n", err)))
					return nil
				}
			}

			if err := kubemaster.CreateClientAndWaitForAPI(kubeconfigs["admin"]); err != nil {
				return err
			}
			// TODO: move Jose server into a separate command or static pod or whatever
			kubemaster.NewDiscoveryEndpoint(params, string(tlsutil.EncodeCertificatePEM(caCert)))

			return nil
		},
	}

	cmd.PersistentFlags().StringVarP(&params.Discovery.ListenIP, "listen-ip", "", "",
		`(optional) IP address to listen on, in case autodetection fails.`)
	cmd.PersistentFlags().StringVarP(&params.Discovery.BearerToken, "token", "", "",
		`(optional) Shared secret used to secure bootstrap. Will be generated and displayed if not provided.`)

	return cmd
}
