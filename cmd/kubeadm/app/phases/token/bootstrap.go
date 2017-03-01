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
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	tokenutil "k8s.io/kubernetes/cmd/kubeadm/app/util/token"
	bootstrapapi "k8s.io/kubernetes/pkg/bootstrap/api"
)

const tokenCreateRetries = 5

// CreateNewToken tries to create a token and fails if one with the same ID already exists
func CreateNewToken(client *clientset.Clientset, d *kubeadmapi.TokenDiscovery, tokenDuration time.Duration, usages []string, description string) error {
	return UpdateOrCreateToken(client, d, true, tokenDuration, usages, description)
}

// UpdateOrCreateToken attempts to update a token with the given ID, or create if it does not already exist.
func UpdateOrCreateToken(client *clientset.Clientset, d *kubeadmapi.TokenDiscovery, failIfExists bool, tokenDuration time.Duration, usages []string, description string) error {
	// Let's make sure the token is valid
	if valid, err := tokenutil.ValidateToken(d); !valid {
		return err
	}
	secretName := fmt.Sprintf("%s%s", bootstrapapi.BootstrapTokenSecretPrefix, d.ID)
	var lastErr error
	for i := 0; i < tokenCreateRetries; i++ {
		secret, err := client.Secrets(metav1.NamespaceSystem).Get(secretName, metav1.GetOptions{})
		if err == nil {
			if failIfExists {
				return fmt.Errorf("a token with id %q already exists", d.ID)
			}
			// Secret with this ID already exists, update it:
			secret.Data = encodeTokenSecretData(d, tokenDuration, usages, description)
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
				Data: encodeTokenSecretData(d, tokenDuration, usages, description),
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

// encodeTokenSecretData takes the token discovery object and an optional duration and returns the .Data for the Secret
func encodeTokenSecretData(d *kubeadmapi.TokenDiscovery, duration time.Duration, usages []string, description string) map[string][]byte {
	data := map[string][]byte{
		bootstrapapi.BootstrapTokenIDKey:     []byte(d.ID),
		bootstrapapi.BootstrapTokenSecretKey: []byte(d.Secret),
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
