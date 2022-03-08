/*
Copyright 2022 The Kubernetes Authors.

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

package credentialprovider

import (
	"github.com/onsi/ginkgo"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

// SIGDescribe annotates the test with the SIG label.
func SIGDescribe(text string, body func()) bool {
	return ginkgo.Describe("[sig-cloud-provider][sig-node][sig-auth] "+text, body)
}

/*
These tests assumes external gcp credential provider is installed and configured on k8s node.
however they won't fail unless in tree gcp credential provider is removed or disabled.
Currently to make sure external credential provider is used for tests both below feature gates needs to be enabled in kubelet
KubeletCredentialProviders, DisableKubeletCloudCredentialProviders
*/

var _ = SIGDescribe("ImageCredentialProvider [Feature: KubeletCredentialProviders]", func() {
	f := framework.NewDefaultFramework("image-credential-provider")
	var podClient *framework.PodClient

	ginkgo.BeforeEach(func() {
		podClient = f.PodClient()
	})

	/*
		Release: v1.24
		Testname: Test kubelet image credential provider
		Description: Create Pod with image from private registry, image credentials fetched from external credential provider by kubelet.
	*/
	ginkgo.It("should be able to create pod with image credentials fetched from external credential provider ", func() {
		privateimage := imageutils.GetConfig(imageutils.AgnhostPrivate)
		name := "pod-auth-image-" + string(uuid.NewUUID())
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: name,
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:            "container-auth-image",
						Image:           privateimage.GetE2EImage(),
						ImagePullPolicy: v1.PullAlways,
					},
				},
			},
		}
		podClient.Create(pod)
		err := e2epod.WaitForPodNameRunningInNamespace(f.ClientSet, pod.Name, f.Namespace.Name)
		framework.ExpectNoError(err)
	})

	/*
		Release: v1.24
		Testname: Test kubelet image credential provider
		Description: Create Pod with unauthenticated image, pod should be created even if kubelet fails to fetch credentials.
	*/
	ginkgo.It("should be able to create pod with unauthenticated image even if failed to get image credentials from external credential provider", func() {
		image := imageutils.GetConfig(imageutils.Agnhost)
		name := "pod-unauth-image-" + string(uuid.NewUUID())
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: name,
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:            "container-unauth-image",
						Image:           image.GetE2EImage(),
						ImagePullPolicy: v1.PullAlways,
					},
				},
			},
		}
		podClient.Create(pod)
		err := e2epod.WaitForPodNameRunningInNamespace(f.ClientSet, pod.Name, f.Namespace.Name)
		framework.ExpectNoError(err)
	})
})
