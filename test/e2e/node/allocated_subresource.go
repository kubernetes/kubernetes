/*
Copyright The Kubernetes Authors.

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
	"net/http"

	"github.com/google/go-cmp/cmp"
	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("Pod Allocated Subresource", framework.WithFeatureGate(features.PodAllocatedSubresource), func() {
	f := framework.NewDefaultFramework("pod-allocated")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	ginkgo.It("should return the allocated pod spec with status cleared", func(ctx context.Context) {
		ginkgo.By("creating a pod")
		podName := "pod-allocated-" + string(uuid.NewUUID())
		pod := e2epod.NewAgnhostPod(f.Namespace.Name, podName, nil, nil, nil)

		podClient := e2epod.NewPodClient(f)
		pod = podClient.CreateSync(ctx, pod)

		ginkgo.By("querying the allocated subresource")
		result := f.ClientSet.CoreV1().RESTClient().Get().
			Namespace(f.Namespace.Name).
			Resource("pods").
			Name(pod.Name).
			SubResource("allocated").
			Do(ctx)

		framework.ExpectNoError(result.Error(), "failed to query allocated subresource")

		statusCode := 0
		result.StatusCode(&statusCode)
		gomega.Expect(statusCode).To(gomega.Equal(http.StatusOK))

		var allocatedPod v1.Pod
		framework.ExpectNoError(result.Into(&allocatedPod), "failed to decode response into pod")

		gomega.Expect(allocatedPod.Name).To(gomega.Equal(pod.Name))
		gomega.Expect(allocatedPod.Namespace).To(gomega.Equal(pod.Namespace))
		if !apiequality.Semantic.DeepEqual(pod.Spec, allocatedPod.Spec) {
			framework.Failf("PodSpec from 'allocated' subresource does not match:\n%s", cmp.Diff(pod.Spec, allocatedPod.Spec))
		}
		gomega.Expect(allocatedPod.Status).To(gomega.BeZero())
	})
})
