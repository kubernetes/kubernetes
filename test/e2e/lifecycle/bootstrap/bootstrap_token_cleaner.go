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
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	bootstrapapi "k8s.io/client-go/tools/bootstrap/token/api"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/lifecycle"
)

var secretNeedClean string
var _ = lifecycle.SIGDescribe("[Feature:BootstrapTokens]", func() {

	var c clientset.Interface

	f := framework.NewDefaultFramework("bootstrap-token-cleaner")

	BeforeEach(func() {
		c = f.ClientSet
	})

	AfterEach(func() {
		if len(secretNeedClean) > 0 {
			By("delete the bootstrap token secret")
			err := c.CoreV1().Secrets(metav1.NamespaceSystem).Delete(secretNeedClean, &metav1.DeleteOptions{})
			secretNeedClean = ""
			Expect(err).NotTo(HaveOccurred())
		}
	})
	It("should delete the token secret when the secret expired", func() {
		By("create a new expired bootstrap token secret")
		tokenId, err := GenerateTokenId()
		Expect(err).NotTo(HaveOccurred())
		tokenSecret, err := GenerateTokenSecret()
		Expect(err).NotTo(HaveOccurred())

		secret := newTokenSecret(tokenId, tokenSecret)
		addSecretExpiration(secret, TimeStringFromNow(-time.Hour))
		_, err = c.CoreV1().Secrets(metav1.NamespaceSystem).Create(secret)

		Expect(err).NotTo(HaveOccurred())

		By("wait for the bootstrap token secret be deleted")
		err = WaitForBootstrapTokenSecretToDisappear(c, tokenId)
		Expect(err).NotTo(HaveOccurred())
	})

	It("should not delete the token secret when the secret is not expired", func() {
		By("create a new expired bootstrap token secret")
		tokenId, err := GenerateTokenId()
		Expect(err).NotTo(HaveOccurred())
		tokenSecret, err := GenerateTokenSecret()
		Expect(err).NotTo(HaveOccurred())
		secret := newTokenSecret(tokenId, tokenSecret)
		addSecretExpiration(secret, TimeStringFromNow(time.Hour))
		_, err = c.CoreV1().Secrets(metav1.NamespaceSystem).Create(secret)
		secretNeedClean = bootstrapapi.BootstrapTokenSecretPrefix + tokenId
		Expect(err).NotTo(HaveOccurred())

		By("wait for the bootstrap token secret not be deleted")
		err = WaitForBootstrapTokenSecretNotDisappear(c, tokenId, 20*time.Second)
		Expect(err).NotTo(HaveOccurred())
	})
})
