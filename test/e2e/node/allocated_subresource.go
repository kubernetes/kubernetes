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
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	helpers "k8s.io/component-helpers/resource"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("Pod Allocated Endpoint", framework.WithFeatureGate(features.KubeletAllocatedPodsEndpoint), func() {
	f := framework.NewDefaultFramework("pod-allocated")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	ginkgo.It("should return all allocated pods with status cleared when listing", func(ctx context.Context) {
		ginkgo.By("creating a pod")
		podName := "pod-allocated-list-" + string(uuid.NewUUID())
		pod := e2epod.NewAgnhostPod(f.Namespace.Name, podName, nil, nil, nil)

		podClient := e2epod.NewPodClient(f)
		pod = podClient.CreateSync(ctx, pod)

		ginkgo.By("querying the allocatedPods list endpoint via node proxy")
		gomega.Expect(pod.Spec.NodeName).ToNot(gomega.BeEmpty())
		result := f.ClientSet.CoreV1().RESTClient().Get().
			Resource("nodes").
			Name(pod.Spec.NodeName).
			SubResource("proxy").
			Suffix("allocatedPods").
			Do(ctx)

		framework.ExpectNoError(result.Error(), "failed to query allocated endpoint list")

		statusCode := 0
		result.StatusCode(&statusCode)
		gomega.Expect(statusCode).To(gomega.Equal(http.StatusOK))

		var allocatedPodList v1.PodList
		framework.ExpectNoError(result.Into(&allocatedPodList), "failed to decode response into pod list")

		var foundPod *v1.Pod
		for _, p := range allocatedPodList.Items {
			if p.Name == pod.Name && p.Namespace == pod.Namespace {
				foundPod = &p
				break
			}
		}
		gomega.Expect(foundPod).ToNot(gomega.BeNil(), "created pod not found in allocated pods list")
		gomega.Expect(foundPod.Status).To(gomega.BeZero())
		if !apiequality.Semantic.DeepEqual(pod.Spec, foundPod.Spec) {
			framework.Failf("PodSpec from 'allocated' endpoint list does not match:\n%s", cmp.Diff(pod.Spec, foundPod.Spec))
		}

		ginkgo.By("deleting the pod")
		podClient.DeleteSync(ctx, pod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)

		ginkgo.By("verifying the pod is no longer in the allocated pods list")
		result = f.ClientSet.CoreV1().RESTClient().Get().
			Resource("nodes").
			Name(pod.Spec.NodeName).
			SubResource("proxy").
			Suffix("allocatedPods").
			Do(ctx)

		framework.ExpectNoError(result.Error(), "failed to query allocated endpoint list after deletion")

		statusCode = 0
		result.StatusCode(&statusCode)
		gomega.Expect(statusCode).To(gomega.Equal(http.StatusOK))

		var allocatedPodListAfterDelete v1.PodList
		framework.ExpectNoError(result.Into(&allocatedPodListAfterDelete), "failed to decode response into pod list after deletion")

		for _, p := range allocatedPodListAfterDelete.Items {
			if p.Name == pod.Name && p.Namespace == pod.Namespace {
				framework.Failf("Pod %s/%s still found in allocated pods list after deletion", pod.Namespace, pod.Name)
			}
		}
	})

	ginkgo.It("should not change allocated pod resources when a resize is deferred", func(ctx context.Context) {
		ginkgo.By("creating a guaranteed pod")
		podName := "pod-allocated-deferred-" + string(uuid.NewUUID())
		pod := e2epod.NewAgnhostPod(f.Namespace.Name, podName, nil, nil, nil)
		pod.Spec.Containers[0].Resources = v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("100m"),
				v1.ResourceMemory: resource.MustParse("100Mi"),
			},
			Limits: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("100m"),
				v1.ResourceMemory: resource.MustParse("100Mi"),
			},
		}

		podClient := e2epod.NewPodClient(f)
		pod = podClient.CreateSync(ctx, pod)

		ginkgo.By("querying the allocatedPods endpoint to get initial state")
		gomega.Expect(pod.Spec.NodeName).ToNot(gomega.BeEmpty())

		initialAllocatedPod := getAllocatedPod(ctx, f, pod.Spec.NodeName, pod.Namespace, pod.Name)

		if !apiequality.Semantic.DeepEqual(pod.Spec, initialAllocatedPod.Spec) {
			framework.Failf("PodSpec from 'allocated' endpoint does not match:\n%s", cmp.Diff(pod.Spec, initialAllocatedPod.Spec))
		}

		ginkgo.By("getting node allocatable CPU")
		node, err := f.ClientSet.CoreV1().Nodes().Get(ctx, pod.Spec.NodeName, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to get node")
		nodeAllocatableCPU := node.Status.Allocatable[v1.ResourceCPU]
		framework.Logf("Node %s allocatable CPU: %v", node.Name, nodeAllocatableCPU.String())

		ginkgo.By("attempting a deferred resize (setting CPU request to node allocatable)")
		pod.Spec.Containers[0].Resources = v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceCPU:    nodeAllocatableCPU,
				v1.ResourceMemory: resource.MustParse("100Mi"),
			},
			Limits: v1.ResourceList{
				v1.ResourceCPU:    nodeAllocatableCPU,
				v1.ResourceMemory: resource.MustParse("100Mi"),
			},
		}

		var updatedPod v1.Pod
		err = f.ClientSet.CoreV1().RESTClient().Put().
			Resource("pods").
			Namespace(pod.Namespace).
			Name(pod.Name).
			SubResource("resize").
			Body(pod).
			Do(ctx).
			Into(&updatedPod)

		framework.ExpectNoError(err, "failed to update pod for resize")

		ginkgo.By("waiting for pod resize status to show deferred")
		framework.ExpectNoError(e2epod.WaitForPodCondition(ctx, f.ClientSet, pod.Namespace, pod.Name, "display pod resize status as deferred", f.Timeouts.PodStart, func(pod *v1.Pod) (bool, error) {
			return helpers.IsPodResizeDeferred(pod), nil
		}))

		ginkgo.By("verifying the allocated pod is unchanged")
		currentAllocatedPod := getAllocatedPod(ctx, f, pod.Spec.NodeName, pod.Namespace, pod.Name)

		if !apiequality.Semantic.DeepEqual(initialAllocatedPod.Spec, currentAllocatedPod.Spec) {
			framework.Failf("PodSpec from 'allocated' endpoint changed after deferred resize:\n%s", cmp.Diff(initialAllocatedPod.Spec, currentAllocatedPod.Spec))
		}
	})
})

func getAllocatedPod(ctx context.Context, f *framework.Framework, nodeName, namespace, name string) *v1.Pod {
	ginkgo.GinkgoHelper()
	result := f.ClientSet.CoreV1().RESTClient().Get().
		Resource("nodes").
		Name(nodeName).
		SubResource("proxy").
		Suffix("allocatedPods", namespace, name).
		Do(ctx)
	framework.ExpectNoError(result.Error(), "failed to query allocated endpoint")

	allocatedPod := &v1.Pod{}
	framework.ExpectNoError(result.Into(allocatedPod), "failed to decode response into pod")

	gomega.Expect(allocatedPod.Name).To(gomega.Equal(name))
	gomega.Expect(allocatedPod.Namespace).To(gomega.Equal(namespace))
	gomega.Expect(allocatedPod.Status).To(gomega.BeZero())
	return allocatedPod
}
