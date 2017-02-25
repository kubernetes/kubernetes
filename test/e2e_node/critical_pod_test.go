/*
Copyright 2016 The Kubernetes Authors.

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

package e2e_node

import (
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubeapi "k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	kubelettypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	criticalPodName    = "critical-pod"
	nonCriticalPodName = "normal-pod"
)

var _ = framework.KubeDescribe("CriticalPod", func() {
	f := framework.NewDefaultFramework("critical-pod-test")

	Context("when we need to admit a critical pod", func() {
		tempSetCurrentKubeletConfig(f, func(initialConfig *componentconfig.KubeletConfiguration) {
			initialConfig.FeatureGates += ", ExperimentalCriticalPodAnnotation=true"
		})
		It("should be able to create and delete a critical pod", func() {
			criticalPod := &v1.Pod{
				TypeMeta: metav1.TypeMeta{
					Kind:       "Pod",
					APIVersion: "v1",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      criticalPodName,
					Namespace: kubeapi.NamespaceSystem,
					Annotations: map[string]string{
						kubelettypes.CriticalPodAnnotationKey: "",
					},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "critical-container",
							Image: framework.GetPauseImageNameForHostArch(),
							Resources: v1.ResourceRequirements{
								// request the entire resource capacity of the node, so that
								// admitting this pod requires the other pod to be preempted
								Requests: getNodeCPUAndMemoryCapacity(f),
							},
						},
					},
				},
			}
			Expect(kubelettypes.IsCriticalPod(criticalPod)).To(BeTrue(), "criticalPod should be a critical pod")

			nonCriticalPod := &v1.Pod{
				TypeMeta: metav1.TypeMeta{
					Kind:       "Pod",
					APIVersion: "v1",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: nonCriticalPodName,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "critical-container",
							Image: framework.GetPauseImageNameForHostArch(),
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									"cpu":    resource.MustParse("100m"),
									"memory": resource.MustParse("100Mi"),
								},
							},
						},
					},
				},
			}
			Expect(kubelettypes.IsCriticalPod(nonCriticalPod)).To(BeFalse(), "nonCriticalPod should not be a critical pod")

			f.PodClient().CreateSync(nonCriticalPod)
			f.PodClientNS(kubeapi.NamespaceSystem).CreateSyncInNamespace(criticalPod, kubeapi.NamespaceSystem)
		})
		AfterEach(func() {
			// Delete Pods
			f.PodClient().DeleteSync(nonCriticalPodName, &metav1.DeleteOptions{}, podDisappearTimeout)
			f.PodClientNS(kubeapi.NamespaceSystem).DeleteSyncInNamespace(criticalPodName, kubeapi.NamespaceSystem, &metav1.DeleteOptions{}, podDisappearTimeout)
			// Log Events
			logPodEvents(f)
			logNodeEvents(f)

		})
	})
})

func getNodeCPUAndMemoryCapacity(f *framework.Framework) v1.ResourceList {
	nodeList, err := f.ClientSet.Core().Nodes().List(metav1.ListOptions{})
	framework.ExpectNoError(err)
	// Assuming that there is only one node, because this is a node e2e test.
	Expect(len(nodeList.Items)).To(Equal(1))
	capacity := nodeList.Items[0].Status.Capacity
	return v1.ResourceList{
		v1.ResourceCPU:    capacity[v1.ResourceCPU],
		v1.ResourceMemory: capacity[v1.ResourceMemory],
	}
}
