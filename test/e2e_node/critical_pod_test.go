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

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2"
	kubeapi "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	kubelettypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/nodefeature"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

const (
	criticalPodName   = "static-critical-pod"
	guaranteedPodName = "guaranteed"
	burstablePodName  = "burstable"
	bestEffortPodName = "best-effort"
)

var _ = SIGDescribe("CriticalPod", framework.WithSerial(), framework.WithDisruptive(), nodefeature.CriticalPod, feature.CriticalPod, func() {
	f := framework.NewDefaultFramework("critical-pod-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	ginkgo.Context("when we need to admit a critical pod", func() {
		ginkgo.It("should be able to create and delete a critical pod", func(ctx context.Context) {
			// because adminssion Priority enable, If the priority class is not found, the Pod is rejected.
			node := getNodeName(ctx, f)
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
				Requests: getNodeCPUAndMemoryCapacity(ctx, f),
			}, node)

			// Create pods, starting with non-critical so that the critical preempts the other pods.
			e2epod.NewPodClient(f).CreateBatch(ctx, []*v1.Pod{nonCriticalBestEffort, nonCriticalBurstable, nonCriticalGuaranteed})
			e2epod.PodClientNS(f, kubeapi.NamespaceSystem).CreateSync(ctx, criticalPod)

			// Check that non-critical pods other than the besteffort have been evicted
			updatedPodList, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).List(ctx, metav1.ListOptions{})
			framework.ExpectNoError(err)
			for _, p := range updatedPodList.Items {
				if p.Name == nonCriticalBestEffort.Name {
					gomega.Expect(p.Status.Phase).To(gomega.Equal(v1.PodRunning), "pod: %v should not be preempted with status: %#v", p.Name, p.Status)
				} else {
					gomega.Expect(p.Status.Phase).To(gomega.Equal(v1.PodSucceeded), "pod: %v should be preempted with status: %#v", p.Name, p.Status)
				}
			}
		})

		f.It("should add DisruptionTarget condition to the preempted pod", func(ctx context.Context) {
			// because adminssion Priority enable, If the priority class is not found, the Pod is rejected.
			node := getNodeName(ctx, f)
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

			criticalPod := getTestPod(true, criticalPodName, v1.ResourceRequirements{
				// request the entire resource capacity of the node, so that
				// admitting this pod requires the other pod to be preempted
				Requests: getNodeCPUAndMemoryCapacity(ctx, f),
			}, node)
			criticalPod.Namespace = kubeapi.NamespaceSystem

			ginkgo.By(fmt.Sprintf("create the non-critical pod %q", klog.KObj(nonCriticalGuaranteed)))
			e2epod.NewPodClient(f).CreateSync(ctx, nonCriticalGuaranteed)

			ginkgo.By(fmt.Sprintf("create the critical pod %q", klog.KObj(criticalPod)))
			e2epod.PodClientNS(f, kubeapi.NamespaceSystem).Create(ctx, criticalPod)

			ginkgo.By(fmt.Sprintf("await for the critical pod %q to be ready", klog.KObj(criticalPod)))
			err := e2epod.WaitForPodNameRunningInNamespace(ctx, f.ClientSet, criticalPod.Name, kubeapi.NamespaceSystem)
			framework.ExpectNoError(err, "Failed to await for the pod to be running: %q", klog.KObj(criticalPod))

			// Check that non-critical pods other than the besteffort have been evicted
			updatedPodList, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).List(ctx, metav1.ListOptions{})
			framework.ExpectNoError(err)
			for _, p := range updatedPodList.Items {
				ginkgo.By(fmt.Sprintf("verify that the non-critical pod %q is preempted and has the DisruptionTarget condition", klog.KObj(&p)))
				gomega.Expect(p.Status.Phase).To(gomega.Equal(v1.PodSucceeded), "pod: %v should be preempted with status: %#v", p.Name, p.Status)
				if condition := e2epod.FindPodConditionByType(&p.Status, v1.DisruptionTarget); condition == nil {
					framework.Failf("pod %q should have the condition: %q, pod status: %v", klog.KObj(&p), v1.DisruptionTarget, p.Status)
				}
			}
		})
		ginkgo.AfterEach(func(ctx context.Context) {
			// Delete Pods
			e2epod.NewPodClient(f).DeleteSync(ctx, guaranteedPodName, metav1.DeleteOptions{}, e2epod.DefaultPodDeletionTimeout)
			e2epod.NewPodClient(f).DeleteSync(ctx, burstablePodName, metav1.DeleteOptions{}, e2epod.DefaultPodDeletionTimeout)
			e2epod.NewPodClient(f).DeleteSync(ctx, bestEffortPodName, metav1.DeleteOptions{}, e2epod.DefaultPodDeletionTimeout)
			e2epod.PodClientNS(f, kubeapi.NamespaceSystem).DeleteSync(ctx, criticalPodName, metav1.DeleteOptions{}, e2epod.DefaultPodDeletionTimeout)
			// Log Events
			logPodEvents(ctx, f)
			logNodeEvents(ctx, f)

		})
	})
})

func getNodeCPUAndMemoryCapacity(ctx context.Context, f *framework.Framework) v1.ResourceList {
	nodeList, err := f.ClientSet.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
	framework.ExpectNoError(err)
	// Assuming that there is only one node, because this is a node e2e test.
	gomega.Expect(nodeList.Items).To(gomega.HaveLen(1))
	capacity := nodeList.Items[0].Status.Allocatable
	return v1.ResourceList{
		v1.ResourceCPU:    capacity[v1.ResourceCPU],
		v1.ResourceMemory: capacity[v1.ResourceMemory],
	}
}

func getNodeName(ctx context.Context, f *framework.Framework) string {
	nodeList, err := f.ClientSet.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
	framework.ExpectNoError(err)
	// Assuming that there is only one node, because this is a node e2e test.
	gomega.Expect(nodeList.Items).To(gomega.HaveLen(1))
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

		if !kubelettypes.IsCriticalPod(pod) {
			framework.Failf("pod %q should be a critical pod", pod.Name)
		}
	} else {
		if kubelettypes.IsCriticalPod(pod) {
			framework.Failf("pod %q should not be a critical pod", pod.Name)
		}
	}
	return pod
}
