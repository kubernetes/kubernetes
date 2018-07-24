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

package node

import (
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	bootstraputil "k8s.io/client-go/tools/bootstrap/token/util"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
)

// TODO(mattmoyer): Move CreateNewTokens, UpdateOrCreateTokens out of this package to client-go for a generic abstraction and client for a Bootstrap Token

// CreateNewTokens tries to create a token and fails if one with the same ID already exists
func CreateNewTokens(client clientset.Interface, tokens []kubeadmapi.BootstrapToken) error {
	return UpdateOrCreateTokens(client, true, tokens)
}

// UpdateOrCreateTokens attempts to update a token with the given ID, or create if it does not already exist.
func UpdateOrCreateTokens(client clientset.Interface, failIfExists bool, tokens []kubeadmapi.BootstrapToken) error {

	for _, token := range tokens {

		secretName := bootstraputil.BootstrapTokenSecretName(token.Token.ID)
		secret, err := client.CoreV1().Secrets(metav1.NamespaceSystem).Get(secretName, metav1.GetOptions{})
		if secret != nil && err == nil && failIfExists {
			return fmt.Errorf("a token with id %q already exists", token.Token.ID)
		}

		updatedOrNewSecret := token.ToSecret()
		// Try to create or update the token with an exponential backoff
		err = apiclient.TryRunCommand(func() error {
			if err := apiclient.CreateOrUpdateSecret(client, updatedOrNewSecret); err != nil {
				return fmt.Errorf("failed to create or update bootstrap token with name %s: %v", secretName, err)
			}
			return nil
		}, 5)
		if err != nil {
			return err
		}
	}
	return nil
}
