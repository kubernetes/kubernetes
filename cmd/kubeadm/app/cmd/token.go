/*
Copyright 2014 The Kubernetes Authors.

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
	"encoding/json"
	"fmt"
	"io"
	"path"

	"github.com/spf13/cobra"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/pkg/api"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
)

func NewCmdToken(out io.Writer) *cobra.Command {
	var skipPreFlight bool
	cmd := &cobra.Command{
		Use:   "token",
		Short: "Manage bootstrap tokens",
		Run: func(cmd *cobra.Command, args []string) {
			err := RunToken(out, cmd, skipPreFlight)
			kubeadmutil.CheckErr(err)
		},
	}

	cmd.PersistentFlags().BoolVar(
		&skipPreFlight, "skip-preflight-checks", false,
		"skip preflight checks normally run before modifying the system",
	)

	return cmd
}

func RunToken(out io.Writer, cmd *cobra.Command, skipPreFlight bool) error {
	if !skipPreFlight {
		fmt.Println("Running pre-flight checks")
		// TODO
	} else {
		fmt.Println("Skipping pre-flight checks")
	}

	envParams := kubeadmapi.GetEnvParams()
	adminKubeconfig, err := clientcmd.LoadFromFile(path.Join(envParams["kubernetes_dir"], "admin.conf"))
	if err != nil {
		return fmt.Errorf("<cmd/token> failed to load admin kubeconfig [%v]", err)
	}
	adminClientConfig, err := clientcmd.NewDefaultClientConfig(
		*adminKubeconfig,
		&clientcmd.ConfigOverrides{},
	).ClientConfig()
	if err != nil {
		return fmt.Errorf("<cmd/token> failed to create API client configuration [%v]", err)
	}

	client, err := clientset.NewForConfig(adminClientConfig)
	if err != nil {
		return fmt.Errorf("<cmd/token> failed to create API client [%v]", err)
	}
	fmt.Println(client)

	tokenSecret := &kubeadmapi.Secrets{}
	err = kubeadmutil.GenerateToken(tokenSecret)
	if err != nil {
		return err
	}
	fmt.Println(tokenSecret.GivenToken)

	fmt.Println("<cmd/token> Submitting token secret to server.")
	secret := &api.Secret{
		ObjectMeta: api.ObjectMeta{Name: tokenSecret.TokenID},
		Type:       api.SecretTypeOpaque,
		Data:       encodeTokenSecretData(tokenSecret),
	}
	if _, err := client.Secrets(api.NamespaceSystem).Create(secret); err != nil {
		return fmt.Errorf("<cmd/token> failed to create token secret [%v]", err)
	}

	return nil
}

func encodeTokenSecretData(tokenSecret *kubeadmapi.Secrets) map[string][]byte {
	var (
		data = map[string][]byte{}
	)

	data[tokenSecret.TokenID], _ = json.Marshal(tokenSecret)

	return data
}
