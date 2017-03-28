/*
Copyright 2015 The Kubernetes Authors.

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

package common

import (
	"fmt"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	testutils "k8s.io/kubernetes/test/utils"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	probTestContainerName       = "test-webserver"
	probTestInitialDelaySeconds = 15

	defaultObservationTimeout = time.Minute * 2
)

var _ = framework.KubeDescribe("Probing container", func() {
	f := framework.NewDefaultFramework("container-probe")
	var podClient *framework.PodClient
	probe := webserverProbeBuilder{}

	BeforeEach(func() {
		podClient = f.PodClient()
	})

	It("with readiness probe should not be ready before initial delay and never restart [Conformance]", func() {
		p := podClient.Create(makePodSpec(probe.withInitialDelay().build(), nil))
		f.WaitForPodReady(p.Name)

		p, err := podClient.Get(p.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		isReady, err := testutils.PodRunningReady(p)
		framework.ExpectNoError(err)
		Expect(isReady).To(BeTrue(), "pod should be ready")

		// We assume the pod became ready when the container became ready. This
		// is true for a single container pod.
		readyTime, err := getTransitionTimeForReadyCondition(p)
		framework.ExpectNoError(err)
		startedTime, err := getContainerStartedTime(p, probTestContainerName)
		framework.ExpectNoError(err)

		framework.Logf("Container started at %v, pod became ready at %v", startedTime, readyTime)
		initialDelay := probTestInitialDelaySeconds * time.Second
		if readyTime.Sub(startedTime) < initialDelay {
			framework.Failf("Pod became ready before it's %v initial delay", initialDelay)
		}

		restartCount := getRestartCount(p)
		Expect(restartCount == 0).To(BeTrue(), "pod should have a restart count of 0 but got %v", restartCount)
	})

	It("with readiness probe that fails should never be ready and never restart [Conformance]", func() {
		p := podClient.Create(makePodSpec(probe.withFailing().build(), nil))
		Consistently(func() (bool, error) {
			p, err := podClient.Get(p.Name, metav1.GetOptions{})
			if err != nil {
				return false, err
			}
			return v1.IsPodReady(p), nil
		}, 1*time.Minute, 1*time.Second).ShouldNot(BeTrue(), "pod should not be ready")

		p, err := podClient.Get(p.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)

		isReady, err := testutils.PodRunningReady(p)
		Expect(isReady).NotTo(BeTrue(), "pod should be not ready")

		restartCount := getRestartCount(p)
		Expect(restartCount == 0).To(BeTrue(), "pod should have a restart count of 0 but got %v", restartCount)
	})

	It("should be restarted with a exec \"cat /tmp/health\" liveness probe [Conformance]", func() {
		runLivenessTest(f, &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "liveness-exec",
				Labels: map[string]string{"test": "liveness"},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    "liveness",
						Image:   "gcr.io/google_containers/busybox:1.24",
						Command: []string{"/bin/sh", "-c", "echo ok >/tmp/health; sleep 10; rm -rf /tmp/health; sleep 600"},
						LivenessProbe: &v1.Probe{
							Handler: v1.Handler{
								Exec: &v1.ExecAction{
									Command: []string{"cat", "/tmp/health"},
								},
							},
							InitialDelaySeconds: 15,
							FailureThreshold:    1,
						},
					},
				},
			},
		}, 1, defaultObservationTimeout)
	})

	It("should *not* be restarted with a exec \"cat /tmp/health\" liveness probe [Conformance]", func() {
		runLivenessTest(f, &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "liveness-exec",
				Labels: map[string]string{"test": "liveness"},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    "liveness",
						Image:   "gcr.io/google_containers/busybox:1.24",
						Command: []string{"/bin/sh", "-c", "echo ok >/tmp/health; sleep 600"},
						LivenessProbe: &v1.Probe{
							Handler: v1.Handler{
								Exec: &v1.ExecAction{
									Command: []string{"cat", "/tmp/health"},
								},
							},
							InitialDelaySeconds: 15,
							FailureThreshold:    1,
						},
					},
				},
			},
		}, 0, defaultObservationTimeout)
	})

	It("should be restarted with a /healthz http liveness probe [Conformance]", func() {
		runLivenessTest(f, &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "liveness-http",
				Labels: map[string]string{"test": "liveness"},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    "liveness",
						Image:   "gcr.io/google_containers/liveness:e2e",
						Command: []string{"/server"},
						LivenessProbe: &v1.Probe{
							Handler: v1.Handler{
								HTTPGet: &v1.HTTPGetAction{
									Path: "/healthz",
									Port: intstr.FromInt(8080),
								},
							},
							InitialDelaySeconds: 15,
							FailureThreshold:    1,
						},
					},
				},
			},
		}, 1, defaultObservationTimeout)
	})

	// Slow by design (5 min)
	It("should have monotonically increasing restart count [Conformance] [Slow]", func() {
		runLivenessTest(f, &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "liveness-http",
				Labels: map[string]string{"test": "liveness"},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    "liveness",
						Image:   "gcr.io/google_containers/liveness:e2e",
						Command: []string{"/server"},
						LivenessProbe: &v1.Probe{
							Handler: v1.Handler{
								HTTPGet: &v1.HTTPGetAction{
									Path: "/healthz",
									Port: intstr.FromInt(8080),
								},
							},
							InitialDelaySeconds: 5,
							FailureThreshold:    1,
						},
					},
				},
			},
		}, 5, time.Minute*5)
	})

	It("should *not* be restarted with a /healthz http liveness probe [Conformance]", func() {
		runLivenessTest(f, &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "liveness-http",
				Labels: map[string]string{"test": "liveness"},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "liveness",
						Image: "gcr.io/google_containers/nginx-slim:0.7",
						Ports: []v1.ContainerPort{{ContainerPort: 80}},
						LivenessProbe: &v1.Probe{
							Handler: v1.Handler{
								HTTPGet: &v1.HTTPGetAction{
									Path: "/",
									Port: intstr.FromInt(80),
								},
							},
							InitialDelaySeconds: 15,
							TimeoutSeconds:      5,
							FailureThreshold:    1,
						},
					},
				},
			},
		}, 0, defaultObservationTimeout)
	})

	It("should be restarted with a docker exec liveness probe with timeout [Conformance]", func() {
		// TODO: enable this test once the default exec handler supports timeout.
		Skip("The default exec handler, dockertools.NativeExecHandler, does not support timeouts due to a limitation in the Docker Remote API")
		runLivenessTest(f, &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "liveness-exec",
				Labels: map[string]string{"test": "liveness"},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    "liveness",
						Image:   "gcr.io/google_containers/busybox:1.24",
						Command: []string{"/bin/sh", "-c", "sleep 600"},
						LivenessProbe: &v1.Probe{
							Handler: v1.Handler{
								Exec: &v1.ExecAction{
									Command: []string{"/bin/sh", "-c", "sleep 10"},
								},
							},
							InitialDelaySeconds: 15,
							TimeoutSeconds:      1,
							FailureThreshold:    1,
						},
					},
				},
			},
		}, 1, defaultObservationTimeout)
	})
})

func getContainerStartedTime(p *v1.Pod, containerName string) (time.Time, error) {
	for _, status := range p.Status.ContainerStatuses {
		if status.Name != containerName {
			continue
		}
		if status.State.Running == nil {
			return time.Time{}, fmt.Errorf("Container is not running")
		}
		return status.State.Running.StartedAt.Time, nil
	}
	return time.Time{}, fmt.Errorf("cannot find container named %q", containerName)
}

func getTransitionTimeForReadyCondition(p *v1.Pod) (time.Time, error) {
	for _, cond := range p.Status.Conditions {
		if cond.Type == v1.PodReady {
			return cond.LastTransitionTime.Time, nil
		}
	}
	return time.Time{}, fmt.Errorf("No ready condition can be found for pod")
}

func getRestartCount(p *v1.Pod) int {
	count := 0
	for _, containerStatus := range p.Status.ContainerStatuses {
		count += int(containerStatus.RestartCount)
	}
	return count
}

func makePodSpec(readinessProbe, livenessProbe *v1.Probe) *v1.Pod {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "test-webserver-" + string(uuid.NewUUID())},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:           probTestContainerName,
					Image:          "gcr.io/google_containers/test-webserver:e2e",
					LivenessProbe:  livenessProbe,
					ReadinessProbe: readinessProbe,
				},
			},
		},
	}
	return pod
}

type webserverProbeBuilder struct {
	failing      bool
	initialDelay bool
}

func (b webserverProbeBuilder) withFailing() webserverProbeBuilder {
	b.failing = true
	return b
}

func (b webserverProbeBuilder) withInitialDelay() webserverProbeBuilder {
	b.initialDelay = true
	return b
}

func (b webserverProbeBuilder) build() *v1.Probe {
	probe := &v1.Probe{
		Handler: v1.Handler{
			HTTPGet: &v1.HTTPGetAction{
				Port: intstr.FromInt(80),
				Path: "/",
			},
		},
	}
	if b.initialDelay {
		probe.InitialDelaySeconds = probTestInitialDelaySeconds
	}
	if b.failing {
		probe.HTTPGet.Port = intstr.FromInt(81)
	}
	return probe
}

func runLivenessTest(f *framework.Framework, pod *v1.Pod, expectNumRestarts int, timeout time.Duration) {
	podClient := f.PodClient()
	ns := f.Namespace.Name
	Expect(pod.Spec.Containers).NotTo(BeEmpty())
	containerName := pod.Spec.Containers[0].Name
	// At the end of the test, clean up by removing the pod.
	defer func() {
		By("deleting the pod")
		podClient.Delete(pod.Name, metav1.NewDeleteOptions(0))
	}()
	By(fmt.Sprintf("Creating pod %s in namespace %s", pod.Name, ns))
	podClient.Create(pod)

	// Wait until the pod is not pending. (Here we need to check for something other than
	// 'Pending' other than checking for 'Running', since when failures occur, we go to
	// 'Terminated' which can cause indefinite blocking.)
	framework.ExpectNoError(framework.WaitForPodNotPending(f.ClientSet, ns, pod.Name),
		fmt.Sprintf("starting pod %s in namespace %s", pod.Name, ns))
	framework.Logf("Started pod %s in namespace %s", pod.Name, ns)

	// Check the pod's current state and verify that restartCount is present.
	By("checking the pod's current state and verifying that restartCount is present")
	pod, err := podClient.Get(pod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err, fmt.Sprintf("getting pod %s in namespace %s", pod.Name, ns))
	initialRestartCount := v1.GetExistingContainerStatus(pod.Status.ContainerStatuses, containerName).RestartCount
	framework.Logf("Initial restart count of pod %s is %d", pod.Name, initialRestartCount)

	// Wait for the restart state to be as desired.
	deadline := time.Now().Add(timeout)
	lastRestartCount := initialRestartCount
	observedRestarts := int32(0)
	for start := time.Now(); time.Now().Before(deadline); time.Sleep(2 * time.Second) {
		pod, err = podClient.Get(pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, fmt.Sprintf("getting pod %s", pod.Name))
		restartCount := v1.GetExistingContainerStatus(pod.Status.ContainerStatuses, containerName).RestartCount
		if restartCount != lastRestartCount {
			framework.Logf("Restart count of pod %s/%s is now %d (%v elapsed)",
				ns, pod.Name, restartCount, time.Since(start))
			if restartCount < lastRestartCount {
				framework.Failf("Restart count should increment monotonically: restart cont of pod %s/%s changed from %d to %d",
					ns, pod.Name, lastRestartCount, restartCount)
			}
		}
		observedRestarts = restartCount - initialRestartCount
		if expectNumRestarts > 0 && int(observedRestarts) >= expectNumRestarts {
			// Stop if we have observed more than expectNumRestarts restarts.
			break
		}
		lastRestartCount = restartCount
	}

	// If we expected 0 restarts, fail if observed any restart.
	// If we expected n restarts (n > 0), fail if we observed < n restarts.
	if (expectNumRestarts == 0 && observedRestarts > 0) || (expectNumRestarts > 0 &&
		int(observedRestarts) < expectNumRestarts) {
		framework.Failf("pod %s/%s - expected number of restarts: %d, found restarts: %d",
			ns, pod.Name, expectNumRestarts, observedRestarts)
	}
}
