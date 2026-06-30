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
