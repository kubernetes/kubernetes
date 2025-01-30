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
	"context"

	"github.com/pkg/errors"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	bootstraputil "k8s.io/cluster-bootstrap/token/util"

	bootstraptokenv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/bootstraptoken/v1"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
)

// CreateNewTokens tries to create a token and fails if one with the same ID already exists
func CreateNewTokens(client clientset.Interface, tokens []bootstraptokenv1.BootstrapToken) error {
	return UpdateOrCreateTokens(client, true, tokens)
}

// UpdateOrCreateTokens attempts to update a token with the given ID, or create if it does not already exist.
func UpdateOrCreateTokens(client clientset.Interface, failIfExists bool, tokens []bootstraptokenv1.BootstrapToken) error {

	secretsClient := client.CoreV1().Secrets(metav1.NamespaceSystem)

	for _, token := range tokens {

		secretName := bootstraputil.BootstrapTokenSecretName(token.Token.ID)
		secret, err := secretsClient.Get(context.Background(), secretName, metav1.GetOptions{})
		if secret != nil && err == nil && failIfExists {
			return errors.Errorf("a token with id %q already exists", token.Token.ID)
		}

		updatedOrNewSecret := bootstraptokenv1.BootstrapTokenToSecret(&token)

		var lastError error
		err = wait.PollUntilContextTimeout(
			context.Background(),
			kubeadmconstants.KubernetesAPICallRetryInterval,
			kubeadmapi.GetActiveTimeouts().KubernetesAPICall.Duration,
			true, func(_ context.Context) (bool, error) {
				if err := apiclient.CreateOrUpdate(secretsClient, updatedOrNewSecret); err != nil {
					lastError = errors.Wrapf(err, "failed to create or update bootstrap token with name %s", secretName)
					return false, nil
				}
				return true, nil
			})
		if err != nil {
			return lastError
		}
	}
	return nil
}
