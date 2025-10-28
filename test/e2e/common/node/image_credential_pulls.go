/*
Copyright 2025 The Kubernetes Authors.

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

package node

import (
	"context"
	"path"

	"github.com/onsi/ginkgo/v2"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/features"
	admissionapi "k8s.io/pod-security-admission/api"

	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eregistry "k8s.io/kubernetes/test/e2e/framework/registry"
)

var _ = SIGDescribe("Ensure Credential Pulled Images", func() {
	f := framework.NewDefaultFramework("ensure-credential-pulled-images")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	framework.Describe("pulling images with credentials", feature.KubeletEnsureSecretPulledImages, framework.WithFeatureGate(features.KubeletEnsureSecretPulledImages), framework.WithSerial(), func() {
		var testImage string
		var testSecret *v1.Secret
		var testNode string
		ginkgo.BeforeEach(func(ctx context.Context) {
			var err error
			registryAddress, _, err := e2eregistry.SetupRegistry(ctx, f, false)
			framework.ExpectNoError(err)
			// this is to wait for the complete removal of all registry pods between tests
			ginkgo.DeferCleanup(func(ctx context.Context) {
				f.DeleteNamespace(ctx, f.Namespace.Name)
			})

			testImage = path.Join(registryAddress, "pause:testing")

			testSecret = e2eregistry.User1DockerSecret(registryAddress)
			testSecret.GenerateName = f.UniqueName
			testSecret, err = f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Create(ctx, testSecret, metav1.CreateOptions{})
			framework.ExpectNoError(err)
			origPod := imagePullTest(ctx, f, testImage, v1.PullIfNotPresent, testSecret, "", v1.PodRunning, false)
			testNode = origPod.Spec.NodeName
		})

		for _, strategy := range []v1.PullPolicy{v1.PullIfNotPresent, v1.PullNever} {
			framework.Context(string(strategy), func() {
				framework.It("pod without PullSecret cannot access previously pulled private image", func(ctx context.Context) {
					_ = imagePullTest(ctx, f, testImage, strategy, nil, testNode, v1.PodPending, true)
				})
				framework.It("pod with invalid PullSecret cannot access previously pulled private image", func(ctx context.Context) {
					invalidSecret, err := f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Create(ctx, &v1.Secret{
						ObjectMeta: metav1.ObjectMeta{GenerateName: f.UniqueName},
						Type:       v1.SecretTypeDockerConfigJson,
						Data: map[string][]byte{
							v1.DockerConfigJsonKey: []byte(`{"auths":{"somerepo.com": {"auth": "aW52YWxpZHVzZXI6aW52YWxpZHBhc3N3b3Jk"}}}`),
						},
					}, metav1.CreateOptions{})
					framework.ExpectNoError(err)

					_ = imagePullTest(ctx, f, testImage, v1.PullIfNotPresent, invalidSecret, testNode, v1.PodPending, true)
				})
				framework.It("pod with the same PullSecret can access previously pulled image", func(ctx context.Context) {
					_ = imagePullTest(ctx, f, testImage, strategy, testSecret, testNode, v1.PodRunning, false)
				})
				framework.It("pod with the same credentials in a different secret can access previously pulled image", func(ctx context.Context) {
					newSecret := testSecret.DeepCopy()
					newSecret.Name = ""
					newSecret.ResourceVersion = ""
					newSecret.GenerateName = f.UniqueName
					newSecret, err := f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Create(ctx, newSecret, metav1.CreateOptions{})
					framework.ExpectNoError(err)
					_ = imagePullTest(ctx, f, testImage, strategy, newSecret, testNode, v1.PodRunning, false)
				})
			})
		}
	})
})
