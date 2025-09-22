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

package node

import (
	"context"
	"fmt"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("Container Readiness During Termination", func() {
	f := framework.NewDefaultFramework("container-readiness-termination")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	var (
		cs        clientset.Interface
		ns        string
		podClient *e2epod.PodClient
	)

	ginkgo.BeforeEach(func(ctx context.Context) {
		cs = f.ClientSet
		ns = f.Namespace.Name
		podClient = e2epod.NewPodClient(f)
	})

	ginkgo.It("should update container readiness when containers die during pod termination", func(ctx context.Context) {
		ginkgo.By("Creating a pod with two containers - one dies quickly, one has long preStop hook")
		podName := "test-pod-readiness-" + string(uuid.NewUUID())
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      podName,
				Namespace: ns,
			},
			Spec: v1.PodSpec{
				TerminationGracePeriodSeconds: func() *int64 { v := int64(50); return &v }(),
				Containers: []v1.Container{
					{
						Name:    "fast-container",
						Image:   "alpine:latest",
						Command: []string{"sh", "-c", "trap 'exit 0' TERM; while true; do sleep 1; done"},
						Ports: []v1.ContainerPort{
							{ContainerPort: 8080},
						},
					},
					{
						Name:    "slow-container",
						Image:   "alpine:latest",
						Command: []string{"sh", "-c", "trap 'exit 0' TERM; while true; do sleep 1; done"},
						Ports: []v1.ContainerPort{
							{ContainerPort: 8081},
						},
						Lifecycle: &v1.Lifecycle{
							PreStop: &v1.LifecycleHandler{
								Exec: &v1.ExecAction{
									Command: []string{"sh", "-c", "sleep 30"},
								},
							},
						},
					},
				},
			},
		}

		ginkgo.By("Creating the pod")
		pod = podClient.Create(ctx, pod)

		ginkgo.By("Waiting for pod to be running")
		err := e2epod.WaitForPodRunningInNamespace(ctx, cs, pod)
		framework.ExpectNoError(err, "pod should be running")

		ginkgo.By("Deleting the pod")
		err = podClient.Delete(ctx, pod.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err, "failed to delete pod")

		ginkgo.By("Monitoring pod status during termination")
		var readinessUpdated bool

		// Monitor pod status for up to 2 minutes
		err = wait.PollUntilContextTimeout(ctx, 2*time.Second, 2*time.Minute, true, func(ctx context.Context) (bool, error) {
			pod, err := podClient.Get(ctx, pod.Name, metav1.GetOptions{})
			if err != nil {
				// If pod is not found, it means it was fully deleted
				// This is expected during termination, so we can stop monitoring
				if errors.IsNotFound(err) {
					framework.Logf("Pod was deleted during monitoring - this is expected during termination")
					return true, nil
				}
				return false, err
			}

			// Check if pod is still terminating
			if pod.DeletionTimestamp == nil {
				return false, fmt.Errorf("pod should be terminating")
			}

			// Count ready containers
			readyContainers := 0
			for _, status := range pod.Status.ContainerStatuses {
				if status.Ready {
					readyContainers++
				}
			}

			framework.Logf("Pod status: %d/%d containers ready", readyContainers, len(pod.Status.ContainerStatuses))

			// The key test: we should see 1/2 containers ready at some point during termination
			if readyContainers == 1 {
				readinessUpdated = true
				framework.Logf("SUCCESS: Readiness updated to %d/2 during termination (expected 1/2)", readyContainers)
				return true, nil
			}

			// If all containers are dead, we're done
			if readyContainers == 0 {
				return true, nil
			}

			return false, nil
		})

		framework.ExpectNoError(err, "failed to monitor pod termination")

		ginkgo.By("Verifying readiness was updated during termination")
		gomega.Expect(readinessUpdated).To(gomega.BeTrueBecause("container readiness should be updated when containers die during termination"))

		// Additional verification: check that the pod eventually gets fully terminated
		ginkgo.By("Waiting for pod to be fully terminated")
		err = wait.PollUntilContextTimeout(ctx, 5*time.Second, 3*time.Minute, true, func(ctx context.Context) (bool, error) {
			_, err := podClient.Get(ctx, pod.Name, metav1.GetOptions{})
			if err != nil {
				// Pod not found means it's fully terminated
				return true, nil
			}
			return false, nil
		})
		framework.ExpectNoError(err, "pod should be fully terminated")
	})

	ginkgo.It("should update readiness when container crashes during termination", func(ctx context.Context) {
		ginkgo.By("Creating a pod with container that will crash")
		podName := "test-pod-liveness-readiness-" + string(uuid.NewUUID())
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      podName,
				Namespace: ns,
			},
			Spec: v1.PodSpec{
				TerminationGracePeriodSeconds: func() *int64 { v := int64(120); return &v }(),
				Containers: []v1.Container{
					{
						Name:  "test-container",
						Image: imageutils.GetE2EImage(imageutils.Agnhost),
						Args:  []string{"sh", "-c", "sleep 10; echo 'Container crashing...'; exit 1"},
						Lifecycle: &v1.Lifecycle{
							PreStop: &v1.LifecycleHandler{
								Exec: &v1.ExecAction{
									Command: []string{"sh", "-c", "sleep 30"},
								},
							},
						},
					},
				},
			},
		}

		ginkgo.By("Creating the pod")
		pod = podClient.Create(ctx, pod)

		ginkgo.By("Waiting for pod to be running")
		err := e2epod.WaitForPodRunningInNamespace(ctx, cs, pod)
		framework.ExpectNoError(err, "pod should be running")

		ginkgo.By("Waiting for container to crash and restart")
		var containerRestarted bool
		err = wait.PollUntilContextTimeout(ctx, 2*time.Second, 1*time.Minute, true, func(ctx context.Context) (bool, error) {
			pod, err := podClient.Get(ctx, pod.Name, metav1.GetOptions{})
			if err != nil {
				return false, err
			}

			status := pod.Status.ContainerStatuses[0]
			if status.RestartCount > 0 {
				containerRestarted = true
				framework.Logf("Container restarted, restart count: %d", status.RestartCount)
				return true, nil
			}

			return false, nil
		})
		framework.ExpectNoError(err, "container should restart after crash")
		gomega.Expect(containerRestarted).To(gomega.BeTrueBecause("container should have restarted"))

		ginkgo.By("Deleting the pod during container restart")
		err = podClient.Delete(ctx, pod.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err, "failed to delete pod")

		ginkgo.By("Monitoring readiness during termination with preStop hook")
		var readinessUpdated bool
		err = wait.PollUntilContextTimeout(ctx, 2*time.Second, 2*time.Minute, true, func(ctx context.Context) (bool, error) {
			pod, err := podClient.Get(ctx, pod.Name, metav1.GetOptions{})
			if err != nil {
				return false, err
			}

			if pod.DeletionTimestamp == nil {
				return false, fmt.Errorf("pod should be terminating")
			}

			readyContainers := 0
			for _, status := range pod.Status.ContainerStatuses {
				if status.Ready {
					readyContainers++
				}
			}

			framework.Logf("Pod status during termination: %d/%d containers ready", readyContainers, len(pod.Status.ContainerStatuses))

			// If container is being killed due to crash, readiness should be updated
			if readyContainers < len(pod.Status.ContainerStatuses) {
				readinessUpdated = true
				framework.Logf("SUCCESS: Readiness updated during container crash termination")
				return true, nil
			}

			return false, nil
		})

		framework.ExpectNoError(err, "failed to monitor pod termination")
		gomega.Expect(readinessUpdated).To(gomega.BeTrueBecause("readiness should be updated during container crash termination"))
	})
})
