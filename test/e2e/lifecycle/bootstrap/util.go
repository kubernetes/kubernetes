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

package bootstrap

import (
	"crypto/rand"
	"encoding/hex"
	"errors"
	"time"

	"k8s.io/api/core/v1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	bootstrapapi "k8s.io/cluster-bootstrap/token/api"
	"k8s.io/kubernetes/test/e2e/framework"
)

func newTokenSecret(tokenID, tokenSecret string) *v1.Secret {
	return &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: metav1.NamespaceSystem,
			Name:      bootstrapapi.BootstrapTokenSecretPrefix + tokenID,
		},
		Type: bootstrapapi.SecretTypeBootstrapToken,
		Data: map[string][]byte{
			bootstrapapi.BootstrapTokenIDKey:           []byte(tokenID),
			bootstrapapi.BootstrapTokenSecretKey:       []byte(tokenSecret),
			bootstrapapi.BootstrapTokenUsageSigningKey: []byte("true"),
		},
	}
}

func GenerateTokenId() (string, error) {
	tokenID, err := randBytes(TokenIDBytes)
	if err != nil {
		return "", err
	}
	return tokenID, nil
}
func GenerateTokenSecret() (string, error) {
	tokenSecret, err := randBytes(TokenSecretBytes)
	if err != nil {
		return "", err
	}
	return tokenSecret, err
}

func randBytes(length int) (string, error) {
	b := make([]byte, length)
	_, err := rand.Read(b)
	if err != nil {
		return "", err
	}
	return hex.EncodeToString(b), nil
}

func addSecretExpiration(s *v1.Secret, expiration string) {
	s.Data[bootstrapapi.BootstrapTokenExpirationKey] = []byte(expiration)
}

func TimeStringFromNow(delta time.Duration) string {
	return time.Now().Add(delta).Format(time.RFC3339)
}

func WaitforSignedClusterInfoByBootStrapToken(c clientset.Interface, tokenID string) error {

	return wait.Poll(framework.Poll, 2*time.Minute, func() (bool, error) {
		cfgMap, err := c.CoreV1().ConfigMaps(metav1.NamespacePublic).Get(bootstrapapi.ConfigMapClusterInfo, metav1.GetOptions{})
		if err != nil {
			framework.Failf("Failed to get cluster-info configMap: %v", err)
			return false, err
		}
		_, ok := cfgMap.Data[bootstrapapi.JWSSignatureKeyPrefix+tokenID]
		if !ok {
			return false, nil
		}
		return true, nil
	})
}

func WaitForSignedClusterInfoGetUpdatedByBootstrapToken(c clientset.Interface, tokenID string, signedToken string) error {

	return wait.Poll(framework.Poll, 2*time.Minute, func() (bool, error) {
		cfgMap, err := c.CoreV1().ConfigMaps(metav1.NamespacePublic).Get(bootstrapapi.ConfigMapClusterInfo, metav1.GetOptions{})
		if err != nil {
			framework.Failf("Failed to get cluster-info configMap: %v", err)
			return false, err
		}
		updated, ok := cfgMap.Data[bootstrapapi.JWSSignatureKeyPrefix+tokenID]
		if !ok || updated == signedToken {
			return false, nil
		}
		return true, nil
	})
}

func WaitForSignedClusterInfoByBootstrapTokenToDisappear(c clientset.Interface, tokenID string) error {

	return wait.Poll(framework.Poll, 2*time.Minute, func() (bool, error) {
		cfgMap, err := c.CoreV1().ConfigMaps(metav1.NamespacePublic).Get(bootstrapapi.ConfigMapClusterInfo, metav1.GetOptions{})
		if err != nil {
			framework.Failf("Failed to get cluster-info configMap: %v", err)
			return false, err
		}
		_, ok := cfgMap.Data[bootstrapapi.JWSSignatureKeyPrefix+tokenID]
		if ok {
			return false, nil
		}
		return true, nil
	})
}

func WaitForBootstrapTokenSecretToDisappear(c clientset.Interface, tokenID string) error {

	return wait.Poll(framework.Poll, 1*time.Minute, func() (bool, error) {
		_, err := c.CoreV1().Secrets(metav1.NamespaceSystem).Get(bootstrapapi.BootstrapTokenSecretPrefix+tokenID, metav1.GetOptions{})
		if apierrs.IsNotFound(err) {
			return true, nil
		}
		return false, nil
	})
}

func WaitForBootstrapTokenSecretNotDisappear(c clientset.Interface, tokenID string, t time.Duration) error {
	err := wait.Poll(framework.Poll, t, func() (bool, error) {
		secret, err := c.CoreV1().Secrets(metav1.NamespaceSystem).Get(bootstrapapi.BootstrapTokenSecretPrefix+tokenID, metav1.GetOptions{})
		if apierrs.IsNotFound(err) {
			return true, errors.New("secret not exists")
		}
		if secret != nil {
			return false, nil
		}
		return true, err
	})
	if err == wait.ErrWaitTimeout {
		return nil
	}
	return err
}
