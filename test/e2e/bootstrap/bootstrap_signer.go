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

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	bootstrapapi "k8s.io/kubernetes/pkg/bootstrap/api"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/test/e2e/framework"
)

const (
	TokenIDBytes = 3
)

var _ = framework.KubeDescribe("[Feature:BootstrapSigner]", func() {

	var c clientset.Interface

	f := framework.NewDefaultFramework("bootstrap-signer")

	BeforeEach(func() {
		c = f.ClientSet
	})

	It("should sign the new added bootstrap tokens", func() {
		By("create a new bootstrap token secret")
		tokenId, err := generateTokenId()
		Expect(err).NotTo(HaveOccurred())
		secret := newTokenSecret(tokenId, "tokenSecret")
		_, err = c.CoreV1().Secrets(metav1.NamespaceSystem).Create(secret)
		defer func() {
			By("delete the bootstrap token secret")
			err := c.CoreV1().Secrets(metav1.NamespaceSystem).Delete(bootstrapapi.BootstrapTokenSecretPrefix+tokenId, &metav1.DeleteOptions{})
			Expect(err).NotTo(HaveOccurred())
		}()
		Expect(err).NotTo(HaveOccurred())

		By("wait for the bootstrap token secret be signed")
		err = framework.WaitforSignedBootStrapToken(c, tokenId)
		Expect(err).NotTo(HaveOccurred())
	})

	It("should resign the bootstrap tokens when the clusterInfo ConfigMap updated [Serial][Disruptive]", func() {
		By("create a new bootstrap token secret")
		tokenId, err := generateTokenId()
		Expect(err).NotTo(HaveOccurred())
		secret := newTokenSecret(tokenId, "tokenSecret")
		secret, err = c.CoreV1().Secrets(metav1.NamespaceSystem).Create(secret)
		defer func() {
			err := c.CoreV1().Secrets(metav1.NamespaceSystem).Delete(bootstrapapi.BootstrapTokenSecretPrefix+tokenId, &metav1.DeleteOptions{})
			Expect(err).NotTo(HaveOccurred())
		}()

		By("wait for the bootstrap token secret be signed")
		err = framework.WaitforSignedBootStrapToken(c, tokenId)

		cfgMap, err := f.ClientSet.CoreV1().ConfigMaps(metav1.NamespacePublic).Get(bootstrapapi.ConfigMapClusterInfo, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())
		signedToken, ok := cfgMap.Data[bootstrapapi.JWSSignatureKeyPrefix+tokenId]
		Expect(ok).Should(Equal(true))

		By("update the cluster-info ConfigMap")
		originalData := cfgMap.Data[bootstrapapi.KubeConfigKey]
		cfgMap.Data[bootstrapapi.KubeConfigKey] = "updated"
		_, err = f.ClientSet.CoreV1().ConfigMaps(metav1.NamespacePublic).Update(cfgMap)
		Expect(err).NotTo(HaveOccurred())
		defer func() {
			By("update back the cluster-info ConfigMap")
			cfgMap, err = f.ClientSet.CoreV1().ConfigMaps(metav1.NamespacePublic).Get(bootstrapapi.ConfigMapClusterInfo, metav1.GetOptions{})
			Expect(err).NotTo(HaveOccurred())
			cfgMap.Data[bootstrapapi.KubeConfigKey] = originalData
			_, err = f.ClientSet.CoreV1().ConfigMaps(metav1.NamespacePublic).Update(cfgMap)
			Expect(err).NotTo(HaveOccurred())
		}()

		By("wait for signed bootstrap token updated")
		err = framework.WaitForSignedBootstrapTokenToGetUpdated(c, tokenId, signedToken)
		Expect(err).NotTo(HaveOccurred())
	})

	It("delete the signed bootstrap tokens from clusterInfo ConfigMap when bootstrap token is deleted", func() {
		By("create a new bootstrap token secret")
		tokenId, err := generateTokenId()
		Expect(err).NotTo(HaveOccurred())
		secret := newTokenSecret(tokenId, "tokenSecret")
		_, err = c.CoreV1().Secrets(metav1.NamespaceSystem).Create(secret)
		Expect(err).NotTo(HaveOccurred())

		By("wait for the bootstrap secret be signed")
		err = framework.WaitforSignedBootStrapToken(c, tokenId)
		Expect(err).NotTo(HaveOccurred())

		By("delete the bootstrap token secret")
		err = c.CoreV1().Secrets(metav1.NamespaceSystem).Delete(bootstrapapi.BootstrapTokenSecretPrefix+tokenId, &metav1.DeleteOptions{})
		Expect(err).NotTo(HaveOccurred())

		By("wait for the bootstrap token removed from cluster-info ConfigMap")
		err = framework.WaitForSignedBootstrapTokenToDisappear(c, tokenId)
		Expect(err).NotTo(HaveOccurred())
	})
})

func newTokenSecret(tokenID, tokenSecret string) *v1.Secret {
	return &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: metav1.NamespaceSystem,
			Name:      bootstrapapi.BootstrapTokenSecretPrefix + tokenID,
		},
		Type: v1.SecretTypeBootstrapToken,
		Data: map[string][]byte{
			bootstrapapi.BootstrapTokenIDKey:           []byte(tokenID),
			bootstrapapi.BootstrapTokenSecretKey:       []byte(tokenSecret),
			bootstrapapi.BootstrapTokenUsageSigningKey: []byte("true"),
		},
	}
}

func generateTokenId() (string, error) {
	tokenID, err := randBytes(TokenIDBytes)
	if err != nil {
		return "", err
	}
	return tokenID, nil
}

func randBytes(length int) (string, error) {
	b := make([]byte, length)
	_, err := rand.Read(b)
	if err != nil {
		return "", err
	}
	return hex.EncodeToString(b), nil
}
