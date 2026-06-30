//go:build linux

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

package e2enode

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	kubelettypes "k8s.io/kubelet/pkg/types"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
)

// These tests cover KEP-4438: restarting sidecar (restartable init) containers
// that exit on their own during pod termination, before their ordered
// termination turn has arrived.
var _ = SIGDescribe("Restarting sidecar containers during pod termination",
	framework.WithFeatureGate(features.SidecarsRestartableDuringPodTermination),
	func() {
		f := framework.NewDefaultFramework("sidecar-termination-restart")
		f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

		ginkgo.When("a restartable init container exits before its ordered termination turn", func() {
			const (
				regular1 = "regular-1"
				sidecar1 = "sidecar-1"

				// The sidecar exits on its own this long after starting — after the pod
				// has begun terminating (so the exit happens during termination) but
				// before the main container finishes, i.e. before the sidecar's ordered
				// termination turn.
				sidecarSelfExitSeconds = 25
				// How long the main container takes to handle SIGTERM before exiting. This
				// holds the pod in termination long enough for the sidecar to exit and be
				// restarted before main goes away and the sidecar's ordered turn arrives.
				mainTerminationSeconds = 40
				// How long the sidecar takes to handle SIGTERM once its turn arrives.
				sidecarTerminationSeconds = 5
			)
			// Generous grace period: the pod is expected to terminate in roughly
			// mainTerminationSeconds + sidecarTerminationSeconds, comfortably within it.
			gracePeriod := int64(120)

			var podSpec *v1.Pod

			ginkgo.BeforeEach(func() {
				podSpec = &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name: "sidecar-restart-during-termination",
					},
					Spec: v1.PodSpec{
						RestartPolicy: v1.RestartPolicyNever,
						InitContainers: []v1.Container{
							{
								Name:          sidecar1,
								Image:         agnhostImage,
								RestartPolicy: &containerRestartPolicyAlways,
								Command: ExecCommand(sidecar1, execCommand{
									Delay:              sidecarSelfExitSeconds,
									TerminationSeconds: sidecarTerminationSeconds,
									ExitCode:           0,
								}),
							},
						},
						Containers: []v1.Container{
							{
								Name:  regular1,
								Image: agnhostImage,
								// Stays running until termination, then takes
								// mainTerminationSeconds to handle SIGTERM. This holds the pod
								// in termination long enough for the sidecar to exit and be
								// restarted before the sidecar's ordered turn arrives.
								Command: e2epod.GenerateScriptCmd(fmt.Sprintf("trap 'echo Terminating; sleep %d; exit 0' TERM; while true; do sleep 1 & wait $!; done", mainTerminationSeconds)),
							},
						},
					},
				}
				preparePod(podSpec)
			})

			ginkgo.It("should restart the sidecar and still terminate the pod gracefully within its grace period", func(ctx context.Context) {
				client := e2epod.NewPodClient(f)
				pod := client.Create(ctx, podSpec)

				ginkgo.By("running the pod")
				framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod))

				current, err := client.Get(ctx, pod.Name, metav1.GetOptions{})
				framework.ExpectNoError(err)
				original, found := podutil.GetContainerStatus(current.Status.InitContainerStatuses, sidecar1)
				gomega.Expect(found).To(gomega.BeTrue())

				ginkgo.By("deleting the pod")
				framework.ExpectNoError(client.Delete(ctx, pod.Name, metav1.DeleteOptions{GracePeriodSeconds: &gracePeriod}))

				ginkgo.By("observing the sidecar restart in pod status while the application is still running")
				gomega.Eventually(ctx, func() (bool, error) {
					current, err := client.Get(ctx, pod.Name, metav1.GetOptions{})
					if err != nil {
						return false, err
					}
					sidecar, _ := podutil.GetContainerStatus(current.Status.InitContainerStatuses, sidecar1)
					main, _ := podutil.GetContainerStatus(current.Status.ContainerStatuses, regular1)
					return sidecar.RestartCount > original.RestartCount && sidecar.State.Running != nil && main.State.Running != nil, nil
				}).WithTimeout(time.Duration(sidecarSelfExitSeconds+30) * time.Second).WithPolling(time.Second).
					Should(gomega.BeTrueBecause("the sidecar must restart before the application finishes draining"))

				ginkgo.By("waiting for the pod to terminate, well within its grace period")
				// The pod terminates in order (main, then the restarted sidecar) in
				// roughly mainTerminationSeconds + sidecarTerminationSeconds. Requiring it
				// to disappear well before the grace period proves the restarted sidecar
				// was terminated in order rather than reaped by the grace-deadline SIGKILL,
				// and guards the regression where the restart loop could spin forever.
				framework.ExpectNoError(e2epod.WaitForPodNotFoundInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace,
					time.Duration(mainTerminationSeconds+40)*time.Second))
			})

			ginkgo.It("should have a smaller grace period from a later termination request override the earlier one", func(ctx context.Context) {
				client := e2epod.NewPodClient(f)
				pod := client.Create(ctx, podSpec)

				ginkgo.By("running the pod")
				framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod))

				ginkgo.By("deleting the pod with a long grace period")
				framework.ExpectNoError(client.Delete(ctx, pod.Name, metav1.DeleteOptions{GracePeriodSeconds: &gracePeriod}))

				ginkgo.By("waiting for the application to receive SIGTERM under the original grace period")
				gomega.Eventually(ctx, func() (string, error) {
					return e2epod.GetPodLogs(ctx, f.ClientSet, pod.Namespace, pod.Name, regular1)
				}).WithTimeout(20 * time.Second).WithPolling(time.Second).Should(gomega.ContainSubstring("Terminating"))

				runtime, _, err := getCRIClient(ctx)
				framework.ExpectNoError(err)
				ginkgo.By("deleting the pod again with a much smaller grace period")
				shortGracePeriod := int64(5)
				framework.ExpectNoError(client.Delete(ctx, pod.Name, metav1.DeleteOptions{GracePeriodSeconds: &shortGracePeriod}))

				// API deletion alone does not prove that the runtime stopped the pod.
				ginkgo.By("observing all containers stop within the shorter grace period")
				gomega.Eventually(ctx, func() ([]*runtimeapi.Container, error) {
					return runtime.ListContainers(ctx, &runtimeapi.ContainerFilter{
						LabelSelector: map[string]string{kubelettypes.KubernetesPodUIDLabel: string(pod.UID)},
						State:         &runtimeapi.ContainerStateValue{State: runtimeapi.ContainerState_CONTAINER_RUNNING},
					})
				}).WithTimeout(time.Duration(shortGracePeriod+5) * time.Second).WithPolling(time.Second).
					Should(gomega.BeEmpty())
				framework.ExpectNoError(e2epod.WaitForPodNotFoundInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace, 10*time.Second))
			})
		})

		ginkgo.When("the kubelet restarts mid-termination", func() {
			const (
				regular2 = "regular-2"
				sidecar2 = "sidecar-2"

				// The sidecar exits on its own this long after starting -- after the pod
				// has begun terminating but before its ordered termination turn.
				sidecarSelfExitSeconds = 25
				// How long the sidecar takes to handle SIGTERM once its turn arrives.
				sidecarTerminationSeconds = 5
				// How long the main container takes to handle SIGTERM. This keeps the pod
				// terminating long enough to restart the kubelet mid-termination (shortly
				// after the sidecar has self-exited) and still observe both the sidecar
				// restart and the pod's ordered termination.
				mainTerminationSeconds = 60
			)
			// Generous grace period: comfortably covers mainTerminationSeconds +
			// sidecarTerminationSeconds plus kubelet restart overhead.
			gracePeriod := int64(120)

			var podSpec *v1.Pod

			ginkgo.BeforeEach(func() {
				podSpec = &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name: "sidecar-restart-kubelet-restart-mid-termination",
					},
					Spec: v1.PodSpec{
						RestartPolicy: v1.RestartPolicyNever,
						InitContainers: []v1.Container{
							{
								Name:          sidecar2,
								Image:         agnhostImage,
								RestartPolicy: &containerRestartPolicyAlways,
								Command: ExecCommand(sidecar2, execCommand{
									Delay:              sidecarSelfExitSeconds,
									TerminationSeconds: sidecarTerminationSeconds,
									ExitCode:           0,
								}),
							},
						},
						Containers: []v1.Container{
							{
								Name:    regular2,
								Image:   agnhostImage,
								Command: e2epod.GenerateScriptCmd(fmt.Sprintf("trap 'echo Terminating; sleep %d; exit 0' TERM; while true; do sleep 1 & wait $!; done", mainTerminationSeconds)),
							},
						},
					},
				}
				preparePod(podSpec)
			})

			f.It("should still restart the sidecar and terminate the pod within its grace period when the kubelet restarts mid-termination",
				f.WithSerial(), f.WithDisruptive(), func(ctx context.Context) {
					client := e2epod.NewPodClient(f)
					pod := client.Create(ctx, podSpec)

					ginkgo.By("running the pod")
					framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod))

					ginkgo.By("deleting the pod")
					framework.ExpectNoError(client.Delete(ctx, pod.Name, metav1.DeleteOptions{GracePeriodSeconds: &gracePeriod}))

					ginkgo.By("waiting for the application to receive SIGTERM")
					gomega.Eventually(ctx, func() (string, error) {
						return e2epod.GetPodLogs(ctx, f.ClientSet, pod.Namespace, pod.Name, regular2)
					}).WithTimeout(20 * time.Second).WithPolling(time.Second).Should(gomega.ContainSubstring("Terminating"))
					current, err := client.Get(ctx, pod.Name, metav1.GetOptions{})
					framework.ExpectNoError(err)
					original, found := podutil.GetContainerStatus(current.Status.InitContainerStatuses, sidecar2)
					gomega.Expect(found).To(gomega.BeTrue())
					gomega.Expect(original.State.Running).NotTo(gomega.BeNil())
					gomega.Expect(current.DeletionTimestamp).NotTo(gomega.BeNil())
					deadline := current.DeletionTimestamp.Time

					ginkgo.By("stopping kubelet and observing the sidecar exit through CRI")
					runtime, _, err := getCRIClient(ctx)
					framework.ExpectNoError(err)
					startKubelet := mustStopKubelet(ctx, f)
					kubeletStopped := true
					ginkgo.DeferCleanup(func(ctx context.Context) {
						if kubeletStopped {
							startKubelet(ctx)
						}
					})
					containerID := strings.SplitN(original.ContainerID, "://", 2)
					gomega.Expect(containerID).To(gomega.HaveLen(2))
					gomega.Eventually(ctx, func() (runtimeapi.ContainerState, error) {
						status, err := runtime.ContainerStatus(ctx, containerID[1], false)
						if err != nil {
							return runtimeapi.ContainerState_CONTAINER_UNKNOWN, err
						}
						return status.Status.State, nil
					}).WithTimeout(time.Duration(sidecarSelfExitSeconds+15) * time.Second).WithPolling(time.Second).
						Should(gomega.Equal(runtimeapi.ContainerState_CONTAINER_EXITED))

					ginkgo.By("restarting kubelet and observing a new sidecar instance")
					startKubelet(ctx)
					kubeletStopped = false
					gomega.Eventually(ctx, func() (bool, error) {
						current, err := client.Get(ctx, pod.Name, metav1.GetOptions{})
						if err != nil {
							return false, err
						}
						status, _ := podutil.GetContainerStatus(current.Status.InitContainerStatuses, sidecar2)
						return status.RestartCount > original.RestartCount && status.State.Running != nil, nil
					}).WithTimeout(30 * time.Second).WithPolling(time.Second).
						Should(gomega.BeTrueBecause("a fresh kubelet must restart the sidecar that exited while it was stopped"))

					ginkgo.By("waiting for termination within the original deadline")
					framework.ExpectNoError(e2epod.WaitForPodNotFoundInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace,
						time.Until(deadline)+10*time.Second))
				})
		})
	})
