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

	"github.com/onsi/ginkgo/v2"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	bootstrapapi "k8s.io/cluster-bootstrap/token/api"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/lifecycle"
	admissionapi "k8s.io/pod-security-admission/api"
)

const (
	// TokenIDBytes is the length of the byte array to generate tokenID.
	TokenIDBytes = 3

	// TokenSecretBytes is the length of the byte array to generate tokenSecret.
	TokenSecretBytes = 8
)

var _ = lifecycle.SIGDescribe(feature.BootstrapTokens, func() {

	var c clientset.Interface

	f := framework.NewDefaultFramework("bootstrap-signer")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	ginkgo.AfterEach(func(ctx context.Context) {
		if len(secretNeedClean) > 0 {
			ginkgo.By("delete the bootstrap token secret")
			err := c.CoreV1().Secrets(metav1.NamespaceSystem).Delete(ctx, secretNeedClean, metav1.DeleteOptions{})
			framework.ExpectNoError(err)
			secretNeedClean = ""
		}
	})
	ginkgo.BeforeEach(func() {
		c = f.ClientSet
	})

	ginkgo.It("should sign the new added bootstrap tokens", func(ctx context.Context) {
		ginkgo.By("create a new bootstrap token secret")
		tokenID, err := GenerateTokenID()
		framework.ExpectNoError(err)
		secret := newTokenSecret(tokenID, "tokenSecret")
		_, err = c.CoreV1().Secrets(metav1.NamespaceSystem).Create(ctx, secret, metav1.CreateOptions{})
		secretNeedClean = bootstrapapi.BootstrapTokenSecretPrefix + tokenID

		framework.ExpectNoError(err)

		ginkgo.By("wait for the bootstrap token secret be signed")
		err = WaitforSignedClusterInfoByBootStrapToken(c, tokenID)
		framework.ExpectNoError(err)
	})

	f.It("should resign the bootstrap tokens when the clusterInfo ConfigMap updated", f.WithSerial(), f.WithDisruptive(), func(ctx context.Context) {
		ginkgo.By("create a new bootstrap token secret")
		tokenID, err := GenerateTokenID()
		framework.ExpectNoError(err)
		secret := newTokenSecret(tokenID, "tokenSecret")
		_, err = c.CoreV1().Secrets(metav1.NamespaceSystem).Create(ctx, secret, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		secretNeedClean = bootstrapapi.BootstrapTokenSecretPrefix + tokenID

		ginkgo.By("wait for the bootstrap token secret be signed")
		err = WaitforSignedClusterInfoByBootStrapToken(c, tokenID)
		framework.ExpectNoError(err)

		cfgMap, err := f.ClientSet.CoreV1().ConfigMaps(metav1.NamespacePublic).Get(ctx, bootstrapapi.ConfigMapClusterInfo, metav1.GetOptions{})
		framework.ExpectNoError(err)
		signedToken, ok := cfgMap.Data[bootstrapapi.JWSSignatureKeyPrefix+tokenID]
		if !ok {
			framework.Failf("expected signed token with key %q not found in %+v", bootstrapapi.JWSSignatureKeyPrefix+tokenID, cfgMap.Data)
		}

		ginkgo.By("update the cluster-info ConfigMap")
		originalData := cfgMap.Data[bootstrapapi.KubeConfigKey]
		updatedKubeConfig, err := randBytes(20)
		framework.ExpectNoError(err)
		cfgMap.Data[bootstrapapi.KubeConfigKey] = updatedKubeConfig
		_, err = f.ClientSet.CoreV1().ConfigMaps(metav1.NamespacePublic).Update(ctx, cfgMap, metav1.UpdateOptions{})
		framework.ExpectNoError(err)
		defer func() {
			ginkgo.By("update back the cluster-info ConfigMap")
			cfgMap, err = f.ClientSet.CoreV1().ConfigMaps(metav1.NamespacePublic).Get(ctx, bootstrapapi.ConfigMapClusterInfo, metav1.GetOptions{})
			framework.ExpectNoError(err)
			cfgMap.Data[bootstrapapi.KubeConfigKey] = originalData
			_, err = f.ClientSet.CoreV1().ConfigMaps(metav1.NamespacePublic).Update(ctx, cfgMap, metav1.UpdateOptions{})
			framework.ExpectNoError(err)
		}()

		ginkgo.By("wait for signed bootstrap token updated")
		err = WaitForSignedClusterInfoGetUpdatedByBootstrapToken(c, tokenID, signedToken)
		framework.ExpectNoError(err)
	})

	ginkgo.It("should delete the signed bootstrap tokens from clusterInfo ConfigMap when bootstrap token is deleted", func(ctx context.Context) {
		ginkgo.By("create a new bootstrap token secret")
		tokenID, err := GenerateTokenID()
		framework.ExpectNoError(err)
		secret := newTokenSecret(tokenID, "tokenSecret")
		_, err = c.CoreV1().Secrets(metav1.NamespaceSystem).Create(ctx, secret, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("wait for the bootstrap secret be signed")
		err = WaitforSignedClusterInfoByBootStrapToken(c, tokenID)
		framework.ExpectNoError(err)

		ginkgo.By("delete the bootstrap token secret")
		err = c.CoreV1().Secrets(metav1.NamespaceSystem).Delete(ctx, bootstrapapi.BootstrapTokenSecretPrefix+tokenID, metav1.DeleteOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("wait for the bootstrap token removed from cluster-info ConfigMap")
		err = WaitForSignedClusterInfoByBootstrapTokenToDisappear(c, tokenID)
		framework.ExpectNoError(err)
	})
})
