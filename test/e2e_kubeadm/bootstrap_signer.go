/*
Copyright 2019 The Kubernetes Authors.

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

package kubeadm

import (
	"context"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	bootstrapapi "k8s.io/cluster-bootstrap/token/api"
	"k8s.io/kubernetes/test/e2e/framework"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
)

const (
	bootstrapTokensSignerRoleName = "system:controller:bootstrap-signer"
)

var _ = Describe("bootstrap signer", func() {

	var c clientset.Interface

	f := framework.NewDefaultFramework("bootstrap-signer")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	ginkgo.BeforeEach(func() {
		c = f.ClientSet
	})

	ginkgo.It("should be active", func(ctx context.Context) {
		//NB. this is technically implemented a part of the control-plane phase
		//    and more specifically if the controller manager is properly configured,
		//    the bootstrapsigner controller is activated and the system:controller:bootstrap-signer
		//    group will be automatically created
		ExpectRole(f.ClientSet, kubeSystemNamespace, bootstrapTokensSignerRoleName)
	})

	// NB. This test is disruptive as it mutates cluster-info, but it's not tagged
	// as disruptive or serial because we want to run it in the kubeadm e2e test jobs,
	// which only run tests in parallel.
	f.It("should resign the bootstrap tokens when cluster-info changed and remove the signature when the token is deleted", func(ctx context.Context) {
		ginkgo.By("create a new bootstrap token secret")
		tokenID, err := GenerateTokenID()
		framework.ExpectNoError(err)
		secret := newTokenSecret(tokenID, "tokenSecret")
		_, err = c.CoreV1().Secrets(metav1.NamespaceSystem).Create(ctx, secret, metav1.CreateOptions{})
		framework.ExpectNoError(err)

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

		ginkgo.By("wait for signed bootstrap token to be updated")
		err = WaitForSignedClusterInfoGetUpdatedByBootstrapToken(c, tokenID, signedToken)
		framework.ExpectNoError(err)

		ginkgo.By("delete the bootstrap token secret")
		err = c.CoreV1().Secrets(metav1.NamespaceSystem).Delete(ctx, bootstrapapi.BootstrapTokenSecretPrefix+tokenID, metav1.DeleteOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("wait for the bootstrap token to be removed from cluster-info ConfigMap")
		err = WaitForSignedClusterInfoByBootstrapTokenToDisappear(c, tokenID)
		framework.ExpectNoError(err)
	})
})
