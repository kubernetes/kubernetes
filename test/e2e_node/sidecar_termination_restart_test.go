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
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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
								Command: ExecCommand(regular1, execCommand{
									Delay:              3600,
									TerminationSeconds: mainTerminationSeconds,
									ExitCode:           0,
								}),
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

				ginkgo.By("deleting the pod")
				framework.ExpectNoError(client.Delete(ctx, pod.Name, metav1.DeleteOptions{GracePeriodSeconds: &gracePeriod}))

				// The restart during termination is not surfaced through the API: the
				// pod status (and thus RestartCount and the previous-logs endpoint) is
				// not refreshed while the kubelet is blocked terminating the pod. It is
				// observed instead from the on-disk per-instance container logs, where a
				// second instance ("1.log") appearing proves the sidecar was restarted.
				ginkgo.By("observing the sidecar get restarted after it exits during termination")
				gomega.Eventually(ctx, func() (bool, error) {
					return sidecarHasBeenRestartedOnDisk(f.Namespace.Name, pod.Name, string(pod.UID), sidecar1)
				}).WithTimeout(time.Duration(sidecarSelfExitSeconds+30) * time.Second).WithPolling(2 * time.Second).
					Should(gomega.BeTrueBecause("the sidecar should gain a second instance after it exits before its ordered termination turn"))

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

				ginkgo.By("deleting the pod again with a much smaller grace period")
				shortGracePeriod := int64(5)
				framework.ExpectNoError(client.Delete(ctx, pod.Name, metav1.DeleteOptions{GracePeriodSeconds: &shortGracePeriod}))

				// The second, smaller grace period should override the first: the kubelet
				// recomputes the effective deadline from DeletionGracePeriodSeconds on each
				// termination request, so the pod should disappear close to the short
				// grace period rather than waiting out the original long one.
				ginkgo.By("waiting for the pod to terminate within the shorter, overriding grace period")
				framework.ExpectNoError(e2epod.WaitForPodNotFoundInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace,
					time.Duration(shortGracePeriod+5)*time.Second))
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
								Name:  regular2,
								Image: agnhostImage,
								Command: ExecCommand(regular2, execCommand{
									Delay:              3600,
									TerminationSeconds: mainTerminationSeconds,
									ExitCode:           0,
								}),
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

					ginkgo.By("waiting for the sidecar to self-exit during termination, then restarting the kubelet")
					time.Sleep(time.Duration(sidecarSelfExitSeconds+3) * time.Second)
					restartKubelet(ctx, true)

					// As in the core regression test, the restart is not surfaced through the
					// API while the kubelet is blocked terminating the pod, so it is observed
					// from the on-disk per-instance container logs. This also proves the
					// restarted kubelet's fresh SyncTerminatingPod call still restarts a
					// sidecar that was exited (and possibly not yet restarted) when the
					// kubelet went down.
					ginkgo.By("observing the sidecar get restarted on disk despite the kubelet restart")
					gomega.Eventually(ctx, func() (bool, error) {
						return sidecarHasBeenRestartedOnDisk(f.Namespace.Name, pod.Name, string(pod.UID), sidecar2)
					}).WithTimeout(time.Duration(sidecarSelfExitSeconds+60) * time.Second).WithPolling(2 * time.Second).
						Should(gomega.BeTrueBecause("the sidecar should be restarted even though the kubelet restarted while it was exited"))

					ginkgo.By("waiting for the pod to finish terminating in order")
					// The kubelet restart resets the termination grace clock (a documented
					// alpha limitation), so bound the remaining wait by a fresh budget that
					// covers the main container's SIGTERM handling plus the sidecar's, rather
					// than the original grace period measured from the initial deletion.
					terminationBudget := time.Duration(mainTerminationSeconds+sidecarTerminationSeconds+30) * time.Second
					framework.ExpectNoError(e2epod.WaitForPodNotFoundInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace, terminationBudget))
				})
		})
	})

// sidecarHasBeenRestartedOnDisk reports whether the named container has more than
// one on-disk instance log under /var/log/pods, which means it has been
// (re)started at least twice. The kubelet writes per-instance logs as
// "<restartCount>.log", so the presence of any index >= 1 proves a restart. This
// is used instead of the API because pod status is not refreshed during the
// pod's termination window. A not-yet-present directory is reported as "not yet".
func sidecarHasBeenRestartedOnDisk(namespace, podName, podUID, containerName string) (bool, error) {
	dir := filepath.Join("/var/log/pods", fmt.Sprintf("%s_%s_%s", namespace, podName, podUID), containerName)
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		return false, err
	}
	for _, e := range entries {
		name := e.Name()
		if !strings.HasSuffix(name, ".log") {
			continue
		}
		if idx, err := strconv.Atoi(strings.TrimSuffix(name, ".log")); err == nil && idx >= 1 {
			return true, nil
		}
	}
	return false, nil
}
