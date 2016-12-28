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

package util

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	"k8s.io/kubernetes/pkg/api"
	apierrors "k8s.io/kubernetes/pkg/api/errors"
	v1 "k8s.io/kubernetes/pkg/api/v1"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
)

const (
	TokenIDBytes               = 3
	TokenBytes                 = 8
	BootstrapTokenSecretPrefix = "bootstrap-token-"
	DefaultTokenDuration       = time.Duration(8) * time.Hour
	tokenCreateRetries         = 5
)

func RandBytes(length int) (string, error) {
	b := make([]byte, length)
	_, err := rand.Read(b)
	if err != nil {
		return "", err
	}
	return hex.EncodeToString(b), nil
}

func GenerateToken(d *kubeadmapi.TokenDiscovery) error {
	tokenID, err := RandBytes(TokenIDBytes)
	if err != nil {
		return err
	}

	token, err := RandBytes(TokenBytes)
	if err != nil {
		return err
	}

	d.ID = tokenID
	d.Secret = token
	return nil
}

var (
	tokenRegexpString = "^([a-zA-Z0-9]{6})\\.([a-zA-Z0-9]{16})$"
	tokenRegexp       = regexp.MustCompile(tokenRegexpString)
)

func GenerateTokenIfNeeded(d *kubeadmapi.TokenDiscovery) error {
	ok, err := IsTokenValid(d)
	if err != nil {
		return err
	}
	if ok {
		return nil
	}
	if err := GenerateToken(d); err != nil {
		return err
	}

	return nil
}

func ParseToken(s string) (string, string, error) {
	split := tokenRegexp.FindStringSubmatch(s)
	if len(split) != 3 {
		return "", "", fmt.Errorf("token %q was not of form %q", s, tokenRegexpString)
	}
	return split[1], split[2], nil

}

func BearerToken(d *kubeadmapi.TokenDiscovery) string {
	return fmt.Sprintf("%s.%s", d.ID, d.Secret)
}

func IsTokenValid(d *kubeadmapi.TokenDiscovery) (bool, error) {
	if len(d.ID)+len(d.Secret) == 0 {
		return false, nil
	}
	if _, _, err := ParseToken(d.ID + "." + d.Secret); err != nil {
		return false, err
	}
	return true, nil
}

func DiscoveryPort(d *kubeadmapi.TokenDiscovery) int32 {
	if len(d.Addresses) == 0 {
		return kubeadmapiext.DefaultDiscoveryBindPort
	}
	split := strings.Split(d.Addresses[0], ":")
	if len(split) == 1 {
		return kubeadmapiext.DefaultDiscoveryBindPort
	}
	if i, err := strconv.Atoi(split[1]); err != nil {
		return int32(i)
	}
	return kubeadmapiext.DefaultDiscoveryBindPort
}

// UpdateOrCreateToken attempts to update a token with the given ID, or create if it does
// not already exist.
func UpdateOrCreateToken(client *clientset.Clientset, d *kubeadmapi.TokenDiscovery, tokenDuration time.Duration) error {
	secretName := fmt.Sprintf("%s%s", BootstrapTokenSecretPrefix, d.ID)

	var lastErr error
	for i := 0; i < tokenCreateRetries; i++ {
		secret, err := client.Secrets(api.NamespaceSystem).Get(secretName, metav1.GetOptions{})
		if err == nil {
			// Secret with this ID already exists, update it:
			secret.Data = encodeTokenSecretData(d, tokenDuration)
			if _, err := client.Secrets(api.NamespaceSystem).Update(secret); err == nil {
				return nil
			} else {
				lastErr = err
			}
			continue
		}

		// Secret does not already exist:
		if apierrors.IsNotFound(err) {
			secret = &v1.Secret{
				ObjectMeta: v1.ObjectMeta{
					Name: secretName,
				},
				Type: api.SecretTypeBootstrapToken,
				Data: encodeTokenSecretData(d, tokenDuration),
			}
			if _, err := client.Secrets(api.NamespaceSystem).Create(secret); err == nil {
				return nil
			} else {
				lastErr = err
			}

			continue
		}

	}
	return fmt.Errorf("<util/tokens> unable to create bootstrap token after %d attempts [%v]", tokenCreateRetries, lastErr)
}

func encodeTokenSecretData(d *kubeadmapi.TokenDiscovery, duration time.Duration) map[string][]byte {
	var (
		data = map[string][]byte{}
	)

	data["token-id"] = []byte(d.ID)
	data["token-secret"] = []byte(d.Secret)

	data["usage-bootstrap-signing"] = []byte("true")

	if duration > 0 {
		t := time.Now()
		t = t.Add(duration)
		data["expiration"] = []byte(t.Format(time.RFC3339))
	}

	return data
}
