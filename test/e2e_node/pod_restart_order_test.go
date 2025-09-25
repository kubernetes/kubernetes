/*
Copyright 2024 The Kubernetes Authors.

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

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	schedulingv1 "k8s.io/api/scheduling/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	testutils "k8s.io/kubernetes/test/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("Pod Restart with PodStartingOrderByPriority Featuregate", framework.WithSerial(), func() {
	f := framework.NewDefaultFramework("pod-restart-order-by-priority")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	f.Context("simulate node reboot by shutting off kubelet remove containers and turning kubelet back on. Check pods become running after \"node reboot\"", func() {
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			initialConfig.FeatureGates = map[string]bool{
				string(features.PodStartingOrderByPriority): true,
			}
		})

		const (
			podAmount                    = 6
			podTotal                     = podAmount * 3
			pollInterval                 = 1 * time.Second
			podStatusUpdateTimeout       = 30 * time.Second
			nodeStatusUpdateTimeout      = 30 * time.Second
			priorityClassesCreateTimeout = 10 * time.Second
			nodeShutdownGracePeriod      = 30 * time.Second
		)

		var (
			customClassHigh2 = newPriorityClass("high-priority-2", 1000000)
			customClassHigh  = newPriorityClass("high-priority", 1000)
			customClassLow   = newPriorityClass("low-priority", -1000000)
		)

		ginkgo.BeforeEach(func(ctx context.Context) {
			ginkgo.By("Wait for node to be ready")
			gomega.Expect(e2enode.WaitForAllNodesSchedulable(ctx, f.ClientSet, 5*time.Minute)).To(gomega.Succeed())
			// Create custom priority classes
			customClasses := []*schedulingv1.PriorityClass{customClassHigh2, customClassHigh, customClassLow}
			for _, customClass := range customClasses {
				_, err := f.ClientSet.SchedulingV1().PriorityClasses().Create(ctx, customClass, metav1.CreateOptions{})
				if err != nil && !apierrors.IsAlreadyExists(err) {
					framework.ExpectNoError(err)
				}
			}
			gomega.Eventually(ctx, func(ctx context.Context) error {
				for _, customClass := range customClasses {
					_, err := f.ClientSet.SchedulingV1().PriorityClasses().Get(ctx, customClass.Name, metav1.GetOptions{})
					if err != nil {
						return err
					}
				}
				return nil
			}).WithTimeout(priorityClassesCreateTimeout).WithPolling(pollInterval).Should(gomega.Succeed())
		})

		ginkgo.It("should create pods with custom priority classes and restart kubelet and pods", func(ctx context.Context) {
			nodeName := getNodeName(ctx, f)
			nodeSelector := fields.Set{
				"spec.nodeName": nodeName,
			}.AsSelector().String()

			pods := []*v1.Pod{}
			for i := range podAmount {
				newLowPriorityPod := getPodWithPriority(fmt.Sprintf("pod-low-%d", i), nodeName, customClassLow.Name)
				newHighPriorityPod := getPodWithPriority(fmt.Sprintf("pod-high-%d", i), nodeName, customClassHigh.Name)
				newHighPriorityPod2 := getPodWithPriority(fmt.Sprintf("pod-high2-%d", i), nodeName, customClassHigh2.Name)
				pods = append(pods, newLowPriorityPod, newHighPriorityPod, newHighPriorityPod2)
			}
			ginkgo.By("Creating batch of pods with custom priorities")
			e2epod.NewPodClient(f).CreateBatch(ctx, pods)

			ginkgo.By("Verifying batch pods are created and running")
			list, err := e2epod.NewPodClient(f).List(ctx, metav1.ListOptions{
				FieldSelector: nodeSelector,
			})
			if err != nil {
				framework.Failf("Failed to start batch pod: %q", err)
			}
			gomega.Expect(list.Items).To(gomega.HaveLen(podTotal), "the number of pods is not as expected")
			for _, pod := range list.Items {
				if podReady, err := testutils.PodRunningReady(&pod); err != nil || !podReady {
					framework.Failf("Failed to start batch pod: (%v/%v)", pod.Namespace, pod.Name)
				}
			}

			ginkgo.By("stopping the kubelet")
			restartKubelet := mustStopKubelet(ctx, f)

			ginkgo.By("stopping all the local containers - using CRI")
			rs, _, err := getCRIClient()
			framework.ExpectNoError(err)
			sandboxes, err := rs.ListPodSandbox(ctx, &runtimeapi.PodSandboxFilter{})
			framework.ExpectNoError(err)
			for _, sandbox := range sandboxes {
				gomega.Expect(sandbox.Metadata).ToNot(gomega.BeNil())
				ginkgo.By(fmt.Sprintf("deleting pod using CRI: %s/%s -> %s", sandbox.Metadata.Namespace, sandbox.Metadata.Name, sandbox.Id))

				err := rs.RemovePodSandbox(ctx, sandbox.Id)
				framework.ExpectNoError(err)
			}

			ginkgo.By("restarting the kubelet")
			restartKubelet(ctx)

			ginkgo.By("Verifying that all pods are running after shutdown is cancelled")
			// All pods should be running and have restarted at least once
			gomega.Eventually(func() error {
				list, err = e2epod.NewPodClient(f).List(ctx, metav1.ListOptions{
					FieldSelector: nodeSelector,
				})
				if err != nil {
					return err
				}
				gomega.Expect(list.Items).To(gomega.HaveLen(podTotal), "the number of pods is not as expected")

				for _, pod := range list.Items {
					if podReady, err := testutils.PodRunningReady(&pod); err != nil || !podReady {
						framework.Logf("Expecting pod to be running, but it's not currently. Pod: (%v/%v), Pod Status Phase: %q, Pod Status Reason: %q", pod.Namespace, pod.Name, pod.Status.Phase, pod.Status.Reason)
						return fmt.Errorf("pod should be running, phase: %s", pod.Status.Phase)
					}
					// restart count should be at least 1
					if pod.Status.ContainerStatuses[0].RestartCount < 1 {
						framework.Logf("Expecting pod to have restarted at least once, but it has not. Pod: (%v/%v), Restart Count: %d", pod.Namespace, pod.Name, pod.Status.ContainerStatuses[0].RestartCount)
						return fmt.Errorf("pod should have restarted at least once, restart count: %d", pod.Status.ContainerStatuses[0].RestartCount)
					}
				}
				return nil
			}).WithTimeout(podStatusUpdateTimeout + nodeShutdownGracePeriod).WithPolling(pollInterval).Should(gomega.Succeed())
		})
	})
})

func getPodWithPriority(name string, node string, priority string) *v1.Pod {
	gracePeriod := int64(30)
	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    name,
					Image:   busyboxImage,
					Command: []string{"sh", "-c"},
					Args: []string{`
					sleep 9999999 &
					PID=$!
					_term() {
						echo "Caught SIGTERM signal!"
						wait $PID
					}

					trap _term SIGTERM
					wait $PID
					`},
				},
			},
			PriorityClassName:             priority,
			TerminationGracePeriodSeconds: &gracePeriod,
			NodeName:                      node,
			RestartPolicy:                 "Always",
		},
	}
	return pod
}

func newPriorityClass(name string, value int32) *schedulingv1.PriorityClass {
	priority := &schedulingv1.PriorityClass{
		TypeMeta: metav1.TypeMeta{
			Kind:       "PriorityClass",
			APIVersion: "scheduling.k8s.io/v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Value: value,
	}
	return priority
}
