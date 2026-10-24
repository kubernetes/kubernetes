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
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("Shortened Grace Period", framework.WithNodeConformance(), func() {
	f := framework.NewDefaultFramework("shortened-grace-period")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelBaseline

	ginkgo.Context("when a pod is deleted again with a shorter grace period", func() {
		var podClient *e2epod.PodClient
		ginkgo.BeforeEach(func() {
			podClient = e2epod.NewPodClient(f)
		})

		// The container ignores SIGTERM, so the first deletion keeps the kubelet
		// waiting for the whole grace period. A second deletion carrying a much
		// shorter grace period has to preempt that in-flight stop, otherwise the
		// pod would only disappear once the original grace period expired.
		ginkgo.It("should stop the pod before the original grace period elapses", func(ctx context.Context) {
			const (
				gracePeriod      = 100
				gracePeriodShort = 1
				firstStopTimeout = 30 * time.Second
				// shortenedDeleteTimeout is well below gracePeriod so the test fails if the first deletion is not preempted.
				shortenedDeleteTimeout = 30 * time.Second
			)
			podName := "shortened-grace-period-test"
			podClient.CreateSync(ctx, getGracePeriodTestPod(podName, gracePeriod))

			ginkgo.By("deleting the pod with the original grace period")
			err := podClient.Delete(ctx, podName, *metav1.NewDeleteOptions(gracePeriod))
			framework.ExpectNoError(err, "failed to delete pod with the original grace period")

			// Waiting for the first SIGTERM makes the test exercise cancellation of
			// an in-flight runtime stop instead of racing both deletes ahead of it.
			ginkgo.By("waiting for the first container stop to be in flight")
			gomega.Eventually(ctx, func(ctx context.Context) (string, error) {
				logs, err := podClient.GetLogs(podName, &v1.PodLogOptions{Container: podName}).DoRaw(ctx)
				return string(logs), err
			}, firstStopTimeout, time.Second).Should(gomega.ContainSubstring("ignoring SIGTERM"))

			ginkgo.By("deleting the pod again with a shorter grace period")
			start := time.Now()
			podClient.DeleteSync(ctx, podName, *metav1.NewDeleteOptions(gracePeriodShort), shortenedDeleteTimeout)
			framework.Logf("pod disappeared %v after the shortened delete", time.Since(start))
		})

		ginkgo.It("should cancel and rerun a blocking exec PreStop hook", func(ctx context.Context) {
			const (
				gracePeriod      = 100
				gracePeriodShort = 20
				testTimeout      = 30 * time.Second
			)
			podName := "shortened-grace-period-exec-prestop"
			podClient.CreateSync(ctx, getPreStopCountingPod(podName, gracePeriod))
			ginkgo.DeferCleanup(podClient.RemoveFinalizer, podName, shortenedGracePeriodFinalizer)

			ginkgo.By("deleting the pod with the original grace period")
			err := podClient.Delete(ctx, podName, *metav1.NewDeleteOptions(gracePeriod))
			framework.ExpectNoError(err, "failed to delete pod with the original grace period")

			ginkgo.By("waiting for the first PreStop invocation to block")
			waitForPodLog(ctx, podClient, podName, "PRESTOP 1", testTimeout)

			ginkgo.By("deleting the pod again with a shorter grace period")
			start := time.Now()
			err = podClient.Delete(ctx, podName, *metav1.NewDeleteOptions(gracePeriodShort))
			framework.ExpectNoError(err, "failed to delete pod with the shortened grace period")
			err = e2epod.WaitForPodSuccessInNamespaceTimeout(ctx, f.ClientSet, podName, f.Namespace.Name, testTimeout)
			framework.ExpectNoError(err, "pod did not terminate successfully after its PreStop hook was cancelled")
			framework.Logf("pod terminated %v after the shortened delete", time.Since(start))

			pod, err := podClient.Get(ctx, podName, metav1.GetOptions{})
			framework.ExpectNoError(err, "failed to get the terminated pod")
			gomega.Expect(pod.Status.ContainerStatuses).To(gomega.HaveLen(1))
			terminated := pod.Status.ContainerStatuses[0].State.Terminated
			gomega.Expect(terminated).NotTo(gomega.BeNil())
			// Exit code zero proves the replacement hook recorded its result;
			// a missing result exits with two and an extra invocation exits with one.
			gomega.Expect(terminated.ExitCode).To(gomega.Equal(int32(0)))
		})

		ginkgo.It("should cancel a blocking HTTP PreStop hook", func(ctx context.Context) {
			const (
				gracePeriod          = 100
				gracePeriodShort     = 1
				firstPreStopTimeout  = 30 * time.Second
				shortenedDeleteLimit = 30 * time.Second
			)
			podName := "shortened-grace-period-http-prestop"
			podClient.CreateSync(ctx, getBlockingHTTPPreStopPod(podName, gracePeriod))

			ginkgo.By("deleting the pod with the original grace period")
			err := podClient.Delete(ctx, podName, *metav1.NewDeleteOptions(gracePeriod))
			framework.ExpectNoError(err, "failed to delete pod with the original grace period")

			ginkgo.By("waiting for the HTTP PreStop request to block")
			waitForPodLog(ctx, podClient, podName, "GET /shell?cmd=sleep 100000", firstPreStopTimeout)

			ginkgo.By("deleting the pod again with a shorter grace period")
			start := time.Now()
			podClient.DeleteSync(ctx, podName, *metav1.NewDeleteOptions(gracePeriodShort), shortenedDeleteLimit)
			framework.Logf("pod disappeared %v after the shortened delete cancelled its HTTP PreStop hook", time.Since(start))
		})
	})
})

const shortenedGracePeriodFinalizer = "e2e.k8s.io/shortened-grace-period"

func waitForPodLog(ctx context.Context, podClient *e2epod.PodClient, podName, expected string, timeout time.Duration) {
	gomega.Eventually(ctx, func(ctx context.Context) (string, error) {
		logs, err := podClient.GetLogs(podName, &v1.PodLogOptions{Container: podName}).DoRaw(ctx)
		return string(logs), err
	}, timeout, time.Second).Should(gomega.ContainSubstring(expected))
}

// getGracePeriodTestPod returns a pod whose container traps and ignores
// SIGTERM, so that it is only removed once its grace period has expired and
// the container runtime sends SIGKILL.
func getGracePeriodTestPod(name string, gracePeriod int64) *v1.Pod {
	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers: []v1.Container{
				{
					Name:    name,
					Image:   busyboxImage,
					Command: []string{"sh", "-c"},
					Args: []string{`
trap 'echo ignoring SIGTERM' TERM
touch /tmp/sigterm-handler-ready
while true; do sleep 1; done
`},
					ReadinessProbe: &v1.Probe{
						PeriodSeconds: 1,
						ProbeHandler: v1.ProbeHandler{
							Exec: &v1.ExecAction{Command: []string{"sh", "-c", "test -f /tmp/sigterm-handler-ready"}},
						},
					},
				},
			},
			TerminationGracePeriodSeconds: &gracePeriod,
		},
	}
	return pod
}

func getPreStopCountingPod(name string, gracePeriod int64) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:       name,
			Finalizers: []string{shortenedGracePeriodFinalizer},
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Volumes: []v1.Volume{{
				Name:         "shared",
				VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}},
			}},
			Containers: []v1.Container{{
				Name:    name,
				Image:   busyboxImage,
				Command: []string{"sh", "-c"},
				Args: []string{`
term_handler() {
  result=$(cat /shared/result 2>/dev/null || echo 2)
  echo "PRESTOP RESULT $result"
  exit "$result"
}
trap term_handler TERM
touch /tmp/sigterm-handler-ready
last=
while true; do
  current=$(cat /shared/count 2>/dev/null || true)
  if [ -n "$current" ] && [ "$current" != "$last" ]; then
    echo "PRESTOP $current"
    last=$current
  fi
  sleep 1 &
  wait $!
done
`},
				Lifecycle: &v1.Lifecycle{
					PreStop: &v1.LifecycleHandler{
						Exec: &v1.ExecAction{Command: []string{"sh", "-c", `
count=$(cat /shared/count 2>/dev/null || echo 0)
count=$((count+1))
echo "$count" > /shared/count
case "$count" in
  1) sleep 100000 ;;
  2) echo 0 > /shared/result ;;
  *) echo 1 > /shared/result ;;
esac
`}},
					},
				},
				ReadinessProbe: &v1.Probe{
					PeriodSeconds: 1,
					ProbeHandler: v1.ProbeHandler{
						Exec: &v1.ExecAction{Command: []string{"sh", "-c", "test -f /tmp/sigterm-handler-ready"}},
					},
				},
				VolumeMounts: []v1.VolumeMount{{Name: "shared", MountPath: "/shared"}},
			}},
			TerminationGracePeriodSeconds: &gracePeriod,
		},
	}
}

func getBlockingHTTPPreStopPod(name string, gracePeriod int64) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers: []v1.Container{{
				Name:    name,
				Image:   agnhostImage,
				Command: []string{"/agnhost", "netexec", "--http-port=8080", "--udp-port=-1"},
				Lifecycle: &v1.Lifecycle{
					PreStop: &v1.LifecycleHandler{
						HTTPGet: &v1.HTTPGetAction{
							Path: "/shell?cmd=sleep%20100000",
							Port: intstr.FromInt32(8080),
						},
					},
				},
				ReadinessProbe: &v1.Probe{
					PeriodSeconds: 1,
					ProbeHandler: v1.ProbeHandler{
						TCPSocket: &v1.TCPSocketAction{Port: intstr.FromInt32(8080)},
					},
				},
			}},
			TerminationGracePeriodSeconds: &gracePeriod,
		},
	}
}
