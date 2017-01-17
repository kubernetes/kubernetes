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

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	"k8s.io/kubernetes/pkg/api"
	v1 "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
)

const (
	TokenIDBytes               = 3
	TokenSecretBytes           = 8
	BootstrapTokenSecretPrefix = "bootstrap-token-"
	DefaultTokenDuration       = time.Duration(8) * time.Hour
	tokenCreateRetries         = 5
)

var (
	tokenIDRegexpString = "^([a-z0-9]{6})$"
	tokenIDRegexp       = regexp.MustCompile(tokenIDRegexpString)
	tokenRegexpString   = "^([a-z0-9]{6})\\:([a-z0-9]{16})$"
	tokenRegexp         = regexp.MustCompile(tokenRegexpString)
)

func randBytes(length int) (string, error) {
	b := make([]byte, length)
	_, err := rand.Read(b)
	if err != nil {
		return "", err
	}
	return hex.EncodeToString(b), nil
}

// GenerateToken generates a new token with a token ID that is valid as a
// Kubernetes DNS label.
// For more info, see kubernetes/pkg/util/validation/validation.go.
func GenerateToken(d *kubeadmapi.TokenDiscovery) error {
	tokenID, err := randBytes(TokenIDBytes)
	if err != nil {
		return err
	}

	token, err := randBytes(TokenSecretBytes)
	if err != nil {
		return err
	}

	d.ID = strings.ToLower(tokenID)
	d.Secret = strings.ToLower(token)
	return nil
}

// ParseTokenID tries and parse a valid token ID from a string.
// An error is returned in case of failure.
func ParseTokenID(s string) error {
	if !tokenIDRegexp.MatchString(s) {
		return fmt.Errorf("token ID [%q] was not of form [%q]", s, tokenIDRegexpString)
	}
	return nil

}

// ParseToken tries and parse a valid token from a string.
// A token ID and token secret are returned in case of success, an error otherwise.
func ParseToken(s string) (string, string, error) {
	split := tokenRegexp.FindStringSubmatch(s)
	if len(split) != 3 {
		return "", "", fmt.Errorf("token [%q] was not of form [%q]", s, tokenRegexpString)
	}
	return split[1], split[2], nil

}

// BearerToken returns a string representation of the passed token.
func BearerToken(d *kubeadmapi.TokenDiscovery) string {
	return fmt.Sprintf("%s:%s", d.ID, d.Secret)
}

// ValidateToken validates whether a token is well-formed.
// In case it's not, the corresponding error is returned as well.
func ValidateToken(d *kubeadmapi.TokenDiscovery) (bool, error) {
	if _, _, err := ParseToken(d.ID + ":" + d.Secret); err != nil {
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
	// Let's make sure
	if valid, err := ValidateToken(d); !valid {
		return err
	}
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
	return fmt.Errorf(
		"unable to create bootstrap token after %d attempts [%v]",
		tokenCreateRetries,
		lastErr,
	)
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
