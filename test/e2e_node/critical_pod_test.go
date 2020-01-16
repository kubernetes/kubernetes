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

package e2enode

import (
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubeapi "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	kubelettypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
)

const (
	criticalPodName   = "static-critical-pod"
	guaranteedPodName = "guaranteed"
	burstablePodName  = "burstable"
	bestEffortPodName = "best-effort"
)

var _ = framework.KubeDescribe("CriticalPod [Serial] [Disruptive] [NodeFeature:CriticalPod]", func() {
	f := framework.NewDefaultFramework("critical-pod-test")
	ginkgo.Context("when we need to admit a critical pod", func() {
		tempSetCurrentKubeletConfig(f, func(initialConfig *kubeletconfig.KubeletConfiguration) {
			if initialConfig.FeatureGates == nil {
				initialConfig.FeatureGates = make(map[string]bool)
			}
		})

		ginkgo.It("should be able to create and delete a critical pod", func() {
			configEnabled, err := isKubeletConfigEnabled(f)
			framework.ExpectNoError(err)
			if !configEnabled {
				framework.Skipf("unable to run test without dynamic kubelet config enabled.")
			}
			// because adminssion Priority enable, If the priority class is not found, the Pod is rejected.
			node := getNodeName(f)
			// Define test pods
			nonCriticalGuaranteed := getTestPod(false, guaranteedPodName, v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("100m"),
					v1.ResourceMemory: resource.MustParse("100Mi"),
				},
				Limits: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("100m"),
					v1.ResourceMemory: resource.MustParse("100Mi"),
				},
			}, node)
			nonCriticalBurstable := getTestPod(false, burstablePodName, v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("100m"),
					v1.ResourceMemory: resource.MustParse("100Mi"),
				},
			}, node)
			nonCriticalBestEffort := getTestPod(false, bestEffortPodName, v1.ResourceRequirements{}, node)
			criticalPod := getTestPod(true, criticalPodName, v1.ResourceRequirements{
				// request the entire resource capacity of the node, so that
				// admitting this pod requires the other pod to be preempted
				Requests: getNodeCPUAndMemoryCapacity(f),
			}, node)

			// Create pods, starting with non-critical so that the critical preempts the other pods.
			f.PodClient().CreateBatch([]*v1.Pod{nonCriticalBestEffort, nonCriticalBurstable, nonCriticalGuaranteed})
			f.PodClientNS(kubeapi.NamespaceSystem).CreateSync(criticalPod)

			// Check that non-critical pods other than the besteffort have been evicted
			updatedPodList, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).List(metav1.ListOptions{})
			framework.ExpectNoError(err)
			for _, p := range updatedPodList.Items {
				if p.Name == nonCriticalBestEffort.Name {
					framework.ExpectEqual(p.Status.Phase, v1.PodRunning, fmt.Sprintf("pod: %v should not be preempted with status: %#v", p.Name, p.Status))
				} else {
					framework.ExpectEqual(p.Status.Phase, v1.PodFailed, fmt.Sprintf("pod: %v should be preempted with status: %#v", p.Name, p.Status))
				}
			}
		})
		ginkgo.AfterEach(func() {
			// Delete Pods
			f.PodClient().DeleteSync(guaranteedPodName, &metav1.DeleteOptions{}, framework.DefaultPodDeletionTimeout)
			f.PodClient().DeleteSync(burstablePodName, &metav1.DeleteOptions{}, framework.DefaultPodDeletionTimeout)
			f.PodClient().DeleteSync(bestEffortPodName, &metav1.DeleteOptions{}, framework.DefaultPodDeletionTimeout)
			f.PodClientNS(kubeapi.NamespaceSystem).DeleteSync(criticalPodName, &metav1.DeleteOptions{}, framework.DefaultPodDeletionTimeout)
			// Log Events
			logPodEvents(f)
			logNodeEvents(f)

		})
	})
})

func getNodeCPUAndMemoryCapacity(f *framework.Framework) v1.ResourceList {
	nodeList, err := f.ClientSet.CoreV1().Nodes().List(metav1.ListOptions{})
	framework.ExpectNoError(err)
	// Assuming that there is only one node, because this is a node e2e test.
	framework.ExpectEqual(len(nodeList.Items), 1)
	capacity := nodeList.Items[0].Status.Allocatable
	return v1.ResourceList{
		v1.ResourceCPU:    capacity[v1.ResourceCPU],
		v1.ResourceMemory: capacity[v1.ResourceMemory],
	}
}

func getNodeName(f *framework.Framework) string {
	nodeList, err := f.ClientSet.CoreV1().Nodes().List(metav1.ListOptions{})
	framework.ExpectNoError(err)
	// Assuming that there is only one node, because this is a node e2e test.
	framework.ExpectEqual(len(nodeList.Items), 1)
	return nodeList.Items[0].GetName()
}

func getTestPod(critical bool, name string, resources v1.ResourceRequirements, node string) *v1.Pod {
	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:      "container",
					Image:     imageutils.GetPauseImageName(),
					Resources: resources,
				},
			},
			NodeName: node,
		},
	}
	if critical {
		pod.ObjectMeta.Namespace = kubeapi.NamespaceSystem
		pod.ObjectMeta.Annotations = map[string]string{
			kubelettypes.ConfigSourceAnnotationKey: kubelettypes.FileSource,
		}
		pod.Spec.PriorityClassName = scheduling.SystemNodeCritical

		framework.ExpectEqual(kubelettypes.IsCriticalPod(pod), true, "pod should be a critical pod")
	} else {
		framework.ExpectEqual(kubelettypes.IsCriticalPod(pod), false, "pod should not be a critical pod")
	}
	return pod
}
