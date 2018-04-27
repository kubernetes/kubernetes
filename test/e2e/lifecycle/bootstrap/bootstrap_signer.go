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
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	bootstrapapi "k8s.io/client-go/tools/bootstrap/token/api"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/lifecycle"
)

const (
	TokenIDBytes     = 3
	TokenSecretBytes = 8
)

var _ = lifecycle.SIGDescribe("[Feature:BootstrapTokens]", func() {

	var c clientset.Interface

	f := framework.NewDefaultFramework("bootstrap-signer")
	AfterEach(func() {
		if len(secretNeedClean) > 0 {
			By("delete the bootstrap token secret")
			err := c.CoreV1().Secrets(metav1.NamespaceSystem).Delete(secretNeedClean, &metav1.DeleteOptions{})
			Expect(err).NotTo(HaveOccurred())
			secretNeedClean = ""
		}
	})
	BeforeEach(func() {
		c = f.ClientSet
	})

	It("should sign the new added bootstrap tokens", func() {
		By("create a new bootstrap token secret")
		tokenId, err := GenerateTokenId()
		Expect(err).NotTo(HaveOccurred())
		secret := newTokenSecret(tokenId, "tokenSecret")
		_, err = c.CoreV1().Secrets(metav1.NamespaceSystem).Create(secret)
		secretNeedClean = bootstrapapi.BootstrapTokenSecretPrefix + tokenId

		Expect(err).NotTo(HaveOccurred())

		By("wait for the bootstrap token secret be signed")
		err = WaitforSignedClusterInfoByBootStrapToken(c, tokenId)
		Expect(err).NotTo(HaveOccurred())
	})

	It("should resign the bootstrap tokens when the clusterInfo ConfigMap updated [Serial][Disruptive]", func() {
		By("create a new bootstrap token secret")
		tokenId, err := GenerateTokenId()
		Expect(err).NotTo(HaveOccurred())
		secret := newTokenSecret(tokenId, "tokenSecret")
		secret, err = c.CoreV1().Secrets(metav1.NamespaceSystem).Create(secret)
		secretNeedClean = bootstrapapi.BootstrapTokenSecretPrefix + tokenId

		By("wait for the bootstrap token secret be signed")
		err = WaitforSignedClusterInfoByBootStrapToken(c, tokenId)

		cfgMap, err := f.ClientSet.CoreV1().ConfigMaps(metav1.NamespacePublic).Get(bootstrapapi.ConfigMapClusterInfo, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())
		signedToken, ok := cfgMap.Data[bootstrapapi.JWSSignatureKeyPrefix+tokenId]
		Expect(ok).Should(Equal(true))

		By("update the cluster-info ConfigMap")
		originalData := cfgMap.Data[bootstrapapi.KubeConfigKey]
		updatedKubeConfig, err := randBytes(20)
		Expect(err).NotTo(HaveOccurred())
		cfgMap.Data[bootstrapapi.KubeConfigKey] = updatedKubeConfig
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
		err = WaitForSignedClusterInfoGetUpdatedByBootstrapToken(c, tokenId, signedToken)
		Expect(err).NotTo(HaveOccurred())
	})

	It("should delete the signed bootstrap tokens from clusterInfo ConfigMap when bootstrap token is deleted", func() {
		By("create a new bootstrap token secret")
		tokenId, err := GenerateTokenId()
		Expect(err).NotTo(HaveOccurred())
		secret := newTokenSecret(tokenId, "tokenSecret")
		_, err = c.CoreV1().Secrets(metav1.NamespaceSystem).Create(secret)
		Expect(err).NotTo(HaveOccurred())

		By("wait for the bootstrap secret be signed")
		err = WaitforSignedClusterInfoByBootStrapToken(c, tokenId)
		Expect(err).NotTo(HaveOccurred())

		By("delete the bootstrap token secret")
		err = c.CoreV1().Secrets(metav1.NamespaceSystem).Delete(bootstrapapi.BootstrapTokenSecretPrefix+tokenId, &metav1.DeleteOptions{})
		Expect(err).NotTo(HaveOccurred())

		By("wait for the bootstrap token removed from cluster-info ConfigMap")
		err = WaitForSignedClusterInfoByBootstrapTokenToDisappear(c, tokenId)
		Expect(err).NotTo(HaveOccurred())
	})
})
