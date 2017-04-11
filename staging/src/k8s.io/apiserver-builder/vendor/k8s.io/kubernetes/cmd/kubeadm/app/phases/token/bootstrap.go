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

package token

import (
	"fmt"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/pkg/api/v1"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	tokenutil "k8s.io/kubernetes/cmd/kubeadm/app/util/token"
	bootstrapapi "k8s.io/kubernetes/pkg/bootstrap/api"
)

const tokenCreateRetries = 5

// CreateNewToken tries to create a token and fails if one with the same ID already exists
func CreateNewToken(client *clientset.Clientset, token string, tokenDuration time.Duration, usages []string, description string) error {
	return UpdateOrCreateToken(client, token, true, tokenDuration, usages, description)
}

// UpdateOrCreateToken attempts to update a token with the given ID, or create if it does not already exist.
func UpdateOrCreateToken(client *clientset.Clientset, token string, failIfExists bool, tokenDuration time.Duration, usages []string, description string) error {
	tokenID, tokenSecret, err := tokenutil.ParseToken(token)
	if err != nil {
		return err
	}
	secretName := fmt.Sprintf("%s%s", bootstrapapi.BootstrapTokenSecretPrefix, tokenID)
	var lastErr error
	for i := 0; i < tokenCreateRetries; i++ {
		secret, err := client.Secrets(metav1.NamespaceSystem).Get(secretName, metav1.GetOptions{})
		if err == nil {
			if failIfExists {
				return fmt.Errorf("a token with id %q already exists", tokenID)
			}
			// Secret with this ID already exists, update it:
			secret.Data = encodeTokenSecretData(tokenID, tokenSecret, tokenDuration, usages, description)
			if _, err := client.Secrets(metav1.NamespaceSystem).Update(secret); err == nil {
				return nil
			} else {
				lastErr = err
			}
			continue
		}

		// Secret does not already exist:
		if apierrors.IsNotFound(err) {
			secret = &v1.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name: secretName,
				},
				Type: v1.SecretType(bootstrapapi.SecretTypeBootstrapToken),
				Data: encodeTokenSecretData(tokenID, tokenSecret, tokenDuration, usages, description),
			}
			if _, err := client.Secrets(metav1.NamespaceSystem).Create(secret); err == nil {
				return nil
			} else {
				lastErr = err
			}
			continue
		}

	}
	return fmt.Errorf(
		"unable to create bootstrap token after %d attempts [%v]",
		tokenCreateRetries,
		lastErr,
	)
}

// CreateBootstrapConfigMap creates the public cluster-info ConfigMap
func CreateBootstrapConfigMap(file string) error {
	adminConfig, err := clientcmd.LoadFromFile(file)
	if err != nil {
		return fmt.Errorf("failed to load admin kubeconfig [%v]", err)
	}
	client, err := kubeconfigutil.KubeConfigToClientSet(adminConfig)
	if err != nil {
		return err
	}

	adminCluster := adminConfig.Contexts[adminConfig.CurrentContext].Cluster
	// Copy the cluster from admin.conf to the bootstrap kubeconfig, contains the CA cert and the server URL
	bootstrapConfig := &clientcmdapi.Config{
		Clusters: map[string]*clientcmdapi.Cluster{
			"": adminConfig.Clusters[adminCluster],
		},
	}
	bootstrapBytes, err := clientcmd.Write(*bootstrapConfig)
	if err != nil {
		return err
	}

	bootstrapConfigMap := v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: bootstrapapi.ConfigMapClusterInfo},
		Data: map[string]string{
			bootstrapapi.KubeConfigKey: string(bootstrapBytes),
		},
	}

	if _, err := client.CoreV1().ConfigMaps(metav1.NamespacePublic).Create(&bootstrapConfigMap); err != nil {
		return err
	}
	return nil
}

// encodeTokenSecretData takes the token discovery object and an optional duration and returns the .Data for the Secret
func encodeTokenSecretData(tokenId, tokenSecret string, duration time.Duration, usages []string, description string) map[string][]byte {
	data := map[string][]byte{
		bootstrapapi.BootstrapTokenIDKey:     []byte(tokenId),
		bootstrapapi.BootstrapTokenSecretKey: []byte(tokenSecret),
	}

	if duration > 0 {
		// Get the current time, add the specified duration, and format it accordingly
		durationString := time.Now().Add(duration).Format(time.RFC3339)
		data[bootstrapapi.BootstrapTokenExpirationKey] = []byte(durationString)
	}
	if len(description) > 0 {
		data[bootstrapapi.BootstrapTokenDescriptionKey] = []byte(description)
	}
	for _, usage := range usages {
		// TODO: Validate the usage string here before
		data[bootstrapapi.BootstrapTokenUsagePrefix+usage] = []byte("true")
	}
	return data
}
