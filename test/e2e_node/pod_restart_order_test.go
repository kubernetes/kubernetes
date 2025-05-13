//go:build linux
// +build linux

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
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("Pod Restart with PodStartingOrderByPriority Featuregate [testFocusHere2]", func() {
	f := framework.NewDefaultFramework("pod-restart-order")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	f.Context("restart node and check pods are becoming running [testFocusHere2]", func() {

		const (
			podAmount                    = int32(6)
			pollInterval                 = 1 * time.Second
			podStatusUpdateTimeout       = 30 * time.Second
			priorityClassesCreateTimeout = 10 * time.Second
			nodeShutdownGracePeriod      = 30 * time.Second
		)

		var (
			customClass0 = getPriorityClass("custom-class-0", 100)
			customClass1 = getPriorityClass("custom-class-1", 1000)
			customClass2 = getPriorityClass("custom-class-2", 10000)
			customClass3 = getPriorityClass("custom-class-3", 100000)
			customClass4 = getPriorityClass("custom-class-4", 1000000)
			customClass5 = getPriorityClass("custom-class-5", 10000000)
		)
		/*
			tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *config.KubeletConfiguration) {
				initialConfig.FeatureGates = map[string]bool{
					string(features.PodStartingOrderByPriority): true,
				}
			})
		*/
		var podClient *e2epod.PodClient

		ginkgo.BeforeEach(func(ctx context.Context) {
			ginkgo.By("Wait for the node to be ready")
			waitForNodeReady(ctx)

			ginkgo.By("Create priorityclasses")
			customClasses := []*schedulingv1.PriorityClass{customClass0, customClass1, customClass2, customClass3, customClass4, customClass5}
			// Create priority classes
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
			}, priorityClassesCreateTimeout, pollInterval).Should(gomega.Succeed())

			podClient = e2epod.NewPodClient(f)
		})

		ginkgo.It("Should [testFocusHere2]", func(ctx context.Context) {
			nodeName := getNodeName(ctx, f)
			nodeSelector := fields.Set{
				"spec.nodeName": nodeName,
			}.AsSelector().String()

			// Create pods with custom priority classes
			pods := []*v1.Pod{}
			for i := range podAmount {
				pods = append(pods, getPodWithPriority(fmt.Sprintf("pod-%d", i), nodeName, fmt.Sprintf("custom-class-%d", i)))
			}
			ginkgo.By("creating batch pods")
			podClient.CreateBatch(ctx, pods)

			list, err := podClient.List(ctx, metav1.ListOptions{
				FieldSelector: nodeSelector,
			})

			if err != nil {
				framework.Failf("Failed to start batch pod: %q", err)
			}
			gomega.Expect(list.Items).To(gomega.HaveLen(int(podAmount)), "the number of pods is not as expected")

			ginkgo.By("Verifying batch pods are running")
			for _, pod := range list.Items {
				if podReady, err := testutils.PodRunningReady(&pod); err != nil || !podReady {
					framework.Failf("Failed to start batch pod: (%v/%v)", pod.Namespace, pod.Name)
				}
			}

			ginkgo.By("Emitting shutdown signal")
			err = emitSignalPrepareForShutdown(true)
			framework.ExpectNoError(err)

			ginkgo.By("Verifying that all pods are shutdown")
			// All pod should be shutdown
			gomega.Eventually(func() error {
				list, err = podClient.List(ctx, metav1.ListOptions{
					FieldSelector: nodeSelector,
				})
				if err != nil {
					return err
				}
				gomega.Expect(list.Items).To(gomega.HaveLen(int(podAmount)), "the number of pods is not as expected")

				for _, pod := range list.Items {
					if !isPodShutdown(&pod) {
						framework.Logf("Expecting pod to be shutdown, but it's not currently. Pod: (%v/%v), Pod Status Phase: %q, Pod Status Reason: %q", pod.Namespace, pod.Name, pod.Status.Phase, pod.Status.Reason)
						return fmt.Errorf("pod should be shutdown, phase: %s", pod.Status.Phase)
					}
					podDisruptionCondition := e2epod.FindPodConditionByType(&pod.Status, v1.DisruptionTarget)
					if podDisruptionCondition == nil {
						framework.Failf("pod (%v/%v) should have the condition: %q, pod status: %v", pod.Namespace, pod.Name, v1.DisruptionTarget, pod.Status)
					}
				}
				return nil
			}, podStatusUpdateTimeout+(nodeShutdownGracePeriod), pollInterval).Should(gomega.BeNil())

		})
	})
})

func getPodWithPriority(name string, node string, priority string) *v1.Pod {
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
					Name:  name,
					Image: imageutils.GetPauseImageName(),
				},
			},
			PriorityClassName: priority,
			NodeName:          node,
			RestartPolicy:     "Always",
		},
	}
	return pod
}
