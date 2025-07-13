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

package node

import (
	"context"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var _ = framework.Describe("Container restart policy", framework.WithFeatureGate(features.ContainerRestartRules), func() {
	f := framework.NewDefaultFramework("container-restart-policy")

	ginkgo.Context("when ContainerRestartRules feature is enabled", func() {
		var (
			containerRestartPolicyAlways = v1.ContainerRestartPolicyAlways
			containerRestartPolicyNever  = v1.ContainerRestartPolicyNever
		)

		ginkgo.It("should restart container on rule match", func(ctx context.Context) {
			podName := "restart-rules-exit-code-" + string(uuid.NewUUID())
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: podName,
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					Containers: []v1.Container{
						{
							Name:          "main-container",
							Image:         imageutils.GetE2EImage(imageutils.BusyBox),
							Command:       []string{"/bin/sh", "-c", "exit 42"},
							RestartPolicy: &containerRestartPolicyNever,
							RestartPolicyRules: []v1.ContainerRestartRule{
								{
									Action: v1.ContainerRestartRuleActionRestart,
									ExitCodes: &v1.ContainerRestartRuleOnExitCodes{
										Operator: v1.ContainerRestartRuleOnExitCodesOpIn,
										Values:   []int32{42},
									},
								},
							},
						},
					},
				},
			}

			createAndValidateRestartableContainer(ctx, f, pod, podName, "main-container")
		})

		ginkgo.It("should not restart container on rule mismatch, container restart policy Never", func(ctx context.Context) {
			podName := "restart-rules-no-restart-" + string(uuid.NewUUID())
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: podName,
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					Containers: []v1.Container{
						{
							Name:          "main-container",
							Image:         imageutils.GetE2EImage(imageutils.BusyBox),
							Command:       []string{"/bin/sh", "-c", "exit 1"},
							RestartPolicy: &containerRestartPolicyNever,
							RestartPolicyRules: []v1.ContainerRestartRule{
								{
									Action: v1.ContainerRestartRuleActionRestart,
									ExitCodes: &v1.ContainerRestartRuleOnExitCodes{
										Operator: v1.ContainerRestartRuleOnExitCodesOpIn,
										Values:   []int32{42},
									},
								},
							},
						},
					},
				},
			}

			createAndValidateNonRestartableContainer(ctx, f, pod, podName, "main-container")
		})

		ginkgo.It("should restart container on container-level restart policy Never", func(ctx context.Context) {
			podName := "restart-rules-no-restart-" + string(uuid.NewUUID())
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: podName,
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyAlways,
					Containers: []v1.Container{
						{
							Name:          "main-container",
							Image:         imageutils.GetE2EImage(imageutils.BusyBox),
							Command:       []string{"/bin/sh", "-c", "exit 1"},
							RestartPolicy: &containerRestartPolicyNever,
						},
					},
				},
			}

			createAndValidateNonRestartableContainer(ctx, f, pod, podName, "main-container")
		})

		ginkgo.It("should restart container on container-level restart policy Always", func(ctx context.Context) {
			podName := "restart-rules-no-restart-" + string(uuid.NewUUID())
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: podName,
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					Containers: []v1.Container{
						{
							Name:          "main-container",
							Image:         imageutils.GetE2EImage(imageutils.BusyBox),
							Command:       []string{"/bin/sh", "-c", "exit 1"},
							RestartPolicy: &containerRestartPolicyAlways,
						},
					},
				},
			}

			createAndValidateRestartableContainer(ctx, f, pod, podName, "main-container")
		})

		ginkgo.It("should restart container on pod-level restart policy Always when no container-level restart policy", func(ctx context.Context) {
			podName := "restart-rules-no-match-" + string(uuid.NewUUID())
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: podName,
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyAlways,
					Containers: []v1.Container{
						{
							Name:    "main-container",
							Image:   imageutils.GetE2EImage(imageutils.BusyBox),
							Command: []string{"/bin/sh", "-c", "exit 1"},
						},
					},
				},
			}

			createAndValidateRestartableContainer(ctx, f, pod, podName, "main-container")
		})
	})
})

func createAndValidateRestartableContainer(ctx context.Context, f *framework.Framework, pod *v1.Pod, podName, containerName string) {
	ginkgo.By("Creating the pod")
	e2epod.NewPodClient(f).Create(ctx, pod)

	ginkgo.By("Waiting for the container to restart")
	err := e2epod.WaitForPodCondition(ctx, f.ClientSet, f.Namespace.Name, podName, "container restarted", 10*time.Minute, func(pod *v1.Pod) (bool, error) {
		for _, status := range pod.Status.ContainerStatuses {
			if status.Name == containerName && status.RestartCount > 0 {
				return true, nil
			}
		}
		return false, nil
	})
	framework.ExpectNoError(err, "failed to see container restart")
}

func createAndValidateNonRestartableContainer(ctx context.Context, f *framework.Framework, pod *v1.Pod, podName, containerName string) {
	ginkgo.By("Creating the pod")
	e2epod.NewPodClient(f).Create(ctx, pod)

	ginkgo.By("Waiting for the pod to terminate")
	err := e2epod.WaitTimeoutForPodNoLongerRunningInNamespace(ctx, f.ClientSet, podName, f.Namespace.Name, 10*time.Minute)
	framework.ExpectNoError(err, "failed to wait for pod terminate")

	ginkgo.By("Checking container restart count")
	p, err := e2epod.NewPodClient(f).Get(ctx, podName, metav1.GetOptions{})
	framework.ExpectNoError(err, "failed to get pod")
	for _, status := range p.Status.ContainerStatuses {
		if status.Name == containerName {
			gomega.Expect(status.RestartCount).To(gomega.BeZero())
		}
	}
}
