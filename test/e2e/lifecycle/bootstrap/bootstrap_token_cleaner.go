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
	"context"
	"time"

	"github.com/onsi/ginkgo/v2"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	bootstrapapi "k8s.io/cluster-bootstrap/token/api"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/lifecycle"
	admissionapi "k8s.io/pod-security-admission/api"
)

var secretNeedClean string
var _ = lifecycle.SIGDescribe("[Feature:BootstrapTokens]", func() {

	var c clientset.Interface

	f := framework.NewDefaultFramework("bootstrap-token-cleaner")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

	ginkgo.BeforeEach(func() {
		c = f.ClientSet
	})

	ginkgo.AfterEach(func() {
		if len(secretNeedClean) > 0 {
			ginkgo.By("delete the bootstrap token secret")
			err := c.CoreV1().Secrets(metav1.NamespaceSystem).Delete(context.TODO(), secretNeedClean, metav1.DeleteOptions{})
			secretNeedClean = ""
			framework.ExpectNoError(err)
		}
	})
	ginkgo.It("should delete the token secret when the secret expired", func() {
		ginkgo.By("create a new expired bootstrap token secret")
		tokenID, err := GenerateTokenID()
		framework.ExpectNoError(err)
		tokenSecret, err := GenerateTokenSecret()
		framework.ExpectNoError(err)

		secret := newTokenSecret(tokenID, tokenSecret)
		addSecretExpiration(secret, TimeStringFromNow(-time.Hour))
		_, err = c.CoreV1().Secrets(metav1.NamespaceSystem).Create(context.TODO(), secret, metav1.CreateOptions{})

		framework.ExpectNoError(err)

		ginkgo.By("wait for the bootstrap token secret be deleted")
		err = WaitForBootstrapTokenSecretToDisappear(c, tokenID)
		framework.ExpectNoError(err)
	})

	ginkgo.It("should not delete the token secret when the secret is not expired", func() {
		ginkgo.By("create a new expired bootstrap token secret")
		tokenID, err := GenerateTokenID()
		framework.ExpectNoError(err)
		tokenSecret, err := GenerateTokenSecret()
		framework.ExpectNoError(err)
		secret := newTokenSecret(tokenID, tokenSecret)
		addSecretExpiration(secret, TimeStringFromNow(time.Hour))
		_, err = c.CoreV1().Secrets(metav1.NamespaceSystem).Create(context.TODO(), secret, metav1.CreateOptions{})
		secretNeedClean = bootstrapapi.BootstrapTokenSecretPrefix + tokenID
		framework.ExpectNoError(err)

		ginkgo.By("wait for the bootstrap token secret not be deleted")
		err = WaitForBootstrapTokenSecretNotDisappear(c, tokenID, 20*time.Second)
		framework.ExpectNoError(err)
	})
})
