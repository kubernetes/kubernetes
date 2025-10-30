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

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/features"
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

	f.It("should update container readiness when containers die during pod termination", f.WithNodeConformance(), framework.WithFeatureGate(features.EventedPLEG), func(ctx context.Context) {
		ginkgo.By("Creating a pod with two containers - fast-container (no preStop) and slow-container (long preStop)")
		podName := "test-pod-readiness-" + string(uuid.NewUUID())
		const preStopSleepSeconds = int64(999999) // infinity constant for preStop sleep action
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      podName,
				Namespace: ns,
			},
			Spec: v1.PodSpec{
				TerminationGracePeriodSeconds: func() *int64 { v := int64(preStopSleepSeconds); return &v }(),
				Containers: []v1.Container{
					{
						Name:  "quick-terminating",
						Image: imageutils.GetE2EImage(imageutils.Agnhost),
						Args:  []string{"pause"},
					},
					{
						Name:  "slow-terminating",
						Image: imageutils.GetE2EImage(imageutils.Agnhost),
						Args:  []string{"pause"},
						Lifecycle: &v1.Lifecycle{
							PreStop: &v1.LifecycleHandler{
								Sleep: &v1.SleepAction{Seconds: preStopSleepSeconds},
							},
						},
					},
				},
			},
		}

		ginkgo.By("Creating the pod")
		pod = podClient.Create(ctx, pod)

		ginkgo.DeferCleanup(func(ctx context.Context) {
			ginkgo.By("Cleaning up the test pod")
			err := podClient.Delete(ctx, pod.Name, metav1.DeleteOptions{
				GracePeriodSeconds: func() *int64 { v := int64(0); return &v }(),
			})
			if err != nil && !apierrors.IsNotFound(err) {
				framework.Logf("Failed to delete pod %s: %v", pod.Name, err)
			}
		})

		ginkgo.By("Waiting for pod to be running (readiness not required)")
		err := e2epod.WaitForPodRunningInNamespace(ctx, cs, pod)
		framework.ExpectNoError(err, "pod should be running")

		ginkgo.By("Deleting the pod to start termination")
		err = podClient.Delete(ctx, pod.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err, "failed to delete pod")

		ginkgo.By("Waiting for readiness to reach 1/2 during termination")
		// Wait for quick-terminating-container to become not ready (this should happen quickly)
		gomega.Eventually(ctx, func(ctx context.Context) int {
			pod, err := podClient.Get(ctx, pod.Name, metav1.GetOptions{})
			if err != nil {
				return -1
			}

			if pod.DeletionTimestamp == nil {
				return -1
			}

			readyContainers := 0
			for _, status := range pod.Status.ContainerStatuses {
				if status.Ready {
					readyContainers++
				}
			}

			framework.Logf("Pod status: %d/%d containers ready", readyContainers, len(pod.Status.ContainerStatuses))
			return readyContainers
		}, f.Timeouts.PodStartShort, f.Timeouts.Poll).Should(gomega.Equal(1), "should reach 1/2 containers ready")

		ginkgo.By("Verifying readiness consistently stays at 1/2 for 10 seconds")
		// Verify it STAYS at 1/2 for some time as slow-terminating-container will not terminate due to infinite preStop
		// but fast-terminating-container's status should be updated to not ready and it remains not ready.
		gomega.Consistently(ctx, func(ctx context.Context) (int, error) {
			pod, err := podClient.Get(ctx, pod.Name, metav1.GetOptions{})
			if err != nil {
				return -1, err
			}

			if len(pod.Status.ContainerStatuses) == 0 {
				return -1, nil
			}

			// Count ready containers and log detailed status
			readyContainers := 0
			for _, status := range pod.Status.ContainerStatuses {
				framework.Logf("Container %s: Ready=%v, RestartCount=%d",
					status.Name, status.Ready, status.RestartCount)
				if status.Ready {
					readyContainers++
				}
			}

			framework.Logf("Pod status consistently: %d/%d containers ready", readyContainers, len(pod.Status.ContainerStatuses))
			return readyContainers, nil
		}, 10*time.Second, f.Timeouts.Poll).Should(gomega.Equal(1), "readiness should stay at 1/2 for 10 seconds")
	})
})
