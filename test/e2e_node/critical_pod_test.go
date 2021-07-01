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
	"context"
	"fmt"
	"time"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	kubeapi "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	kubelettypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

const (
	criticalPodName   = "static-critical-pod"
	guaranteedPodName = "guaranteed"
	burstablePodName  = "burstable"
	bestEffortPodName = "best-effort"
)

var _ = SIGDescribe("CriticalPod [Serial] [Disruptive] [NodeFeature:CriticalPod]", func() {
	f := framework.NewDefaultFramework("critical-pod-test")
	ginkgo.Context("when we need to admit a critical pod", func() {
		tempSetCurrentKubeletConfig(f, func(initialConfig *kubeletconfig.KubeletConfiguration) {
			if initialConfig.FeatureGates == nil {
				initialConfig.FeatureGates = make(map[string]bool)
			}
		})

		podMap := make(map[string]*v1.Pod)

		ginkgo.It("should be able to create and delete a critical pod", func() {
			configEnabled, err := isKubeletConfigEnabled(f)
			framework.ExpectNoError(err)
			if !configEnabled {
				e2eskipper.Skipf("unable to run test without dynamic kubelet config enabled.")
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

			ginkgo.By("Creating pods, starting with non-critical so that the critical preempts the other pods.")
			createdPods := f.PodClient().CreateBatch([]*v1.Pod{nonCriticalBestEffort, nonCriticalBurstable, nonCriticalGuaranteed})
			for _, pod := range createdPods {
				podMap[pod.Name] = pod
			}

			ginkgo.By("Creating the critical pod")
			createSyncInNamespaceFromPod(f, criticalPod)

			ginkgo.By("Checking that non-critical pods other than the besteffort have been evicted")
			updatedPodList, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).List(context.TODO(), metav1.ListOptions{})

			ginkgo.By(fmt.Sprintf("Checking that non-critical pods other than the besteffort have been evicted (%d pods)", len(updatedPodList.Items)))
			framework.ExpectNoError(err)
			for _, p := range updatedPodList.Items {
				framework.Logf("pod %q status %v", p.Name, p.Status)
				if p.Name == nonCriticalBestEffort.Name {
					framework.ExpectEqual(p.Status.Phase, v1.PodRunning, fmt.Sprintf("pod: %v should not be preempted with status: %#v", p.Name, p.Status))
				} else {
					framework.ExpectEqual(p.Status.Phase, v1.PodFailed, fmt.Sprintf("pod: %v should be preempted with status: %#v", p.Name, p.Status))
				}
			}
		})
		ginkgo.AfterEach(func() {
			// Delete Pods
			deleteSyncInNamespace(f, criticalPodName, kubeapi.NamespaceSystem, metav1.DeleteOptions{}, framework.DefaultPodDeletionTimeout)
			deletePodsAsync(f, podMap)
			// Log Events
			logPodEvents(f)
			logNodeEvents(f)

		})
	})
})

func createSyncInNamespaceFromPod(f *framework.Framework, pod *v1.Pod) *v1.Pod {
	p := f.PodClientNS(pod.Namespace).Create(pod)
	framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(f.ClientSet, p.Name, p.Namespace, framework.PodStartTimeout))
	// Get the newest pod after it becomes running and ready, some status may change after pod created, such as pod ip.
	p, err := f.PodClientNS(pod.Namespace).Get(context.TODO(), p.Name, metav1.GetOptions{})
	framework.ExpectNoError(err)
	return p
}

func deleteSyncInNamespace(f *framework.Framework, name, namespace string, options metav1.DeleteOptions, timeout time.Duration) {
	err := f.PodClientNS(namespace).Delete(context.TODO(), name, options)
	if err != nil && !apierrors.IsNotFound(err) {
		framework.Failf("Failed to delete pod %q in namespace %q: %v", name, namespace, err)
	}
	gomega.Expect(e2epod.WaitForPodToDisappear(f.ClientSet, namespace, name, labels.Everything(),
		2*time.Second, timeout)).To(gomega.Succeed(), "wait for pod %q in namespace %q  to disappear", name, namespace)
}

func getNodeCPUAndMemoryCapacity(f *framework.Framework) v1.ResourceList {
	nodeList, err := f.ClientSet.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{})
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
	nodeList, err := f.ClientSet.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{})
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
