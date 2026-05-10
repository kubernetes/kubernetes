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

package e2enode

import (
	"context"
	"path"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	internalapi "k8s.io/cri-api/pkg/apis"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	admissionapi "k8s.io/pod-security-admission/api"

	e2ecommonnode "k8s.io/kubernetes/test/e2e/common/node"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eregistry "k8s.io/kubernetes/test/e2e/framework/registry"
)

var _ = SIGDescribe("Ensure Credential Pulled Images", func() {
	f := framework.NewDefaultFramework("ensure-credential-pulled-images")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	var is internalapi.ImageManagerService

	framework.Describe("pulling images with credentials", framework.WithFeatureGate(features.KubeletEnsureSecretPulledImages), framework.WithSerial(), func() {
		var registryAddress string
		var testImage string
		var testImageID string
		var testSecret *v1.Secret
		var testNode string

		ginkgo.BeforeEach(func(ctx context.Context) {
			var err error
			_, is, err = getCRIClient(ctx)
			framework.ExpectNoError(err)

			var registryNodeNames []string
			registryAddress, registryNodeNames, err = e2eregistry.SetupRegistry(ctx, f, true)
			framework.ExpectNoError(err)
			gomega.Expect(registryNodeNames).ToNot(gomega.BeEmpty(), "registry should run on at least one node")
			// this is to wait for the complete removal of all registry pods between tests
			ginkgo.DeferCleanup(func(ctx context.Context) {
				f.DeleteNamespace(ctx, f.Namespace.Name)
			})

			testImage = path.Join(registryAddress, "pause:testing")
			_ = is.RemoveImage(ctx, &runtimeapi.ImageSpec{Image: testImage})

			testSecret = e2eregistry.User1DockerSecret(registryAddress)
			testSecret.GenerateName = f.UniqueName
			testSecret, err = f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Create(ctx, testSecret, metav1.CreateOptions{})
			framework.ExpectNoError(err)
			// Use the registry node for scheduling - in node e2e tests, this is the single test node
			testNode = registryNodeNames[0]
			// PullAlways pull policy will force an ImagePulledRecord to be created
			origPod := e2ecommonnode.ImagePullTest(ctx, f, testImage, v1.PullAlways, testSecret, testNode, v1.PodRunning, false)
			gomega.Expect(origPod.Spec.NodeName).To(gomega.Equal(testNode), "pod should be scheduled on the expected node")

			testImgStatus, err := is.ImageStatus(ctx, &runtimeapi.ImageSpec{Image: testImage}, false)
			framework.ExpectNoError(err)
			testImageID = testImgStatus.Image.Id
		})

		for _, credsPolicy := range []kubeletconfig.ImagePullCredentialsVerificationPolicy{
			kubeletconfig.NeverVerifyPreloadedImages,
			kubeletconfig.AlwaysVerify,
			kubeletconfig.NeverVerifyAllowlistedImages,
			kubeletconfig.NeverVerify,
		} {
			framework.Context(string(credsPolicy), func() {
				tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
					initialConfig.ImagePullCredentialsVerificationPolicy = string(credsPolicy)
				})

				for _, pullPolicy := range []v1.PullPolicy{v1.PullIfNotPresent, v1.PullNever} {
					framework.Context(string(pullPolicy), func() {
						statusOnInvalidCreds := v1.PodPending
						pullingOnInvalidCreds := true
						if credsPolicy == kubeletconfig.NeverVerify {
							statusOnInvalidCreds = v1.PodRunning
							pullingOnInvalidCreds = false
						}

						framework.It("pod without PullSecret accessing previously pulled private image", func(ctx context.Context) {
							_ = e2ecommonnode.ImagePullTest(ctx, f, testImage, pullPolicy, nil, testNode, statusOnInvalidCreds, pullingOnInvalidCreds)
						})
						framework.It("pod with invalid PullSecret accessing previously pulled private image", func(ctx context.Context) {
							invalidSecret, err := f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Create(ctx, &v1.Secret{
								ObjectMeta: metav1.ObjectMeta{GenerateName: f.UniqueName},
								Type:       v1.SecretTypeDockerConfigJson,
								Data: map[string][]byte{
									v1.DockerConfigJsonKey: []byte(`{"auths":{"somerepo.com": {"auth": "aW52YWxpZHVzZXI6aW52YWxpZHBhc3N3b3Jk"}}}`),
								},
							}, metav1.CreateOptions{})
							framework.ExpectNoError(err)

							_ = e2ecommonnode.ImagePullTest(ctx, f, testImage, pullPolicy, invalidSecret, testNode, statusOnInvalidCreds, pullingOnInvalidCreds)
						})
						framework.It("pod with the same PullSecret accessing previously pulled image", func(ctx context.Context) {
							_ = e2ecommonnode.ImagePullTest(ctx, f, testImage, pullPolicy, testSecret, testNode, v1.PodRunning, false)
						})
						framework.It("pod with the same credentials in a different secret accessing previously pulled image", func(ctx context.Context) {
							newSecret := testSecret.DeepCopy()
							newSecret.Name = ""
							newSecret.ResourceVersion = ""
							newSecret.GenerateName = f.UniqueName
							newSecret, err := f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Create(ctx, newSecret, metav1.CreateOptions{})
							framework.ExpectNoError(err)
							_ = e2ecommonnode.ImagePullTest(ctx, f, testImage, pullPolicy, newSecret, testNode, v1.PodRunning, false)
						})

						switch credsPolicy {
						case kubeletconfig.NeverVerifyAllowlistedImages:
							framework.Context("[NeverVerifyAllowListedImages specific tests]", func() {
								tempRemoveImagePulledRecord(f, &testImageID)
								tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
									initialConfig.ImagePullCredentialsVerificationPolicy = string(credsPolicy)
									initialConfig.PreloadedImagesVerificationAllowlist = []string{path.Join(registryAddress, "pause")}
								})

								framework.It("pod without PullSecret accessing previously pulled private image which is allowlisted", func(ctx context.Context) {
									_ = e2ecommonnode.ImagePullTest(ctx, f, testImage, pullPolicy, nil, testNode, v1.PodRunning, false)
								})
							})
						case kubeletconfig.NeverVerifyPreloadedImages:
							framework.Context("[NeverVerifyPreloadedImages specific tests]", func() {
								tempRemoveImagePulledRecord(f, &testImageID)

								framework.It("pod without PullSecret accessing a preloaded image", func(ctx context.Context) {
									_ = e2ecommonnode.ImagePullTest(ctx, f, testImage, pullPolicy, nil, testNode, v1.PodRunning, false)
								})
							})
						} // switch credsPolicy
					}) // framework.Context(pullPolicy)
				} // for pullPolicy
			}) // framework.Context(credsPolicy)
		} // for credsPolicy
	}) // framework.Describe()
})
