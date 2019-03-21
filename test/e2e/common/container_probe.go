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
	"net/url"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/uuid"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/test/e2e/framework"
	testutils "k8s.io/kubernetes/test/utils"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

const (
	probTestContainerName       = "test-webserver"
	probTestInitialDelaySeconds = 15

	defaultObservationTimeout = time.Minute * 4
)

var _ = framework.KubeDescribe("Probing container", func() {
	f := framework.NewDefaultFramework("container-probe")
	var podClient *framework.PodClient
	probe := webserverProbeBuilder{}

	BeforeEach(func() {
		podClient = f.PodClient()
	})

	/*
		Release : v1.9
		Testname: Pod readiness probe, with initial delay
		Description: Create a Pod that is configured with a initial delay set on the readiness probe. Check the Pod Start time to compare to the initial delay. The Pod MUST be ready only after the specified initial delay.
	*/
	framework.ConformanceIt("with readiness probe should not be ready before initial delay and never restart [NodeConformance]", func() {
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

	/*
		Release : v1.9
		Testname: Pod readiness probe, failure
		Description: Create a Pod with a readiness probe that fails consistently. When this Pod is created,
			then the Pod MUST never be ready, never be running and restart count MUST be zero.
	*/
	framework.ConformanceIt("with readiness probe that fails should never be ready and never restart [NodeConformance]", func() {
		p := podClient.Create(makePodSpec(probe.withFailing().build(), nil))
		Consistently(func() (bool, error) {
			p, err := podClient.Get(p.Name, metav1.GetOptions{})
			if err != nil {
				return false, err
			}
			return podutil.IsPodReady(p), nil
		}, 1*time.Minute, 1*time.Second).ShouldNot(BeTrue(), "pod should not be ready")

		p, err := podClient.Get(p.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)

		isReady, err := testutils.PodRunningReady(p)
		Expect(isReady).NotTo(BeTrue(), "pod should be not ready")

		restartCount := getRestartCount(p)
		Expect(restartCount == 0).To(BeTrue(), "pod should have a restart count of 0 but got %v", restartCount)
	})

	/*
		Release : v1.9
		Testname: Pod liveness probe, using local file, restart
		Description: Create a Pod with liveness probe that uses ExecAction handler to cat /temp/health file. The Container deletes the file /temp/health after 10 second, triggering liveness probe to fail. The Pod MUST now be killed and restarted incrementing restart count to 1.
	*/
	framework.ConformanceIt("should be restarted with a exec \"cat /tmp/health\" liveness probe [NodeConformance]", func() {
		runLivenessTest(f, &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "liveness-exec",
				Labels: map[string]string{"test": "liveness"},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    "liveness",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
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

	/*
		Release : v1.9
		Testname: Pod liveness probe, using local file, no restart
		Description:  Pod is created with liveness probe that uses ‘exec’ command to cat /temp/health file. Liveness probe MUST not fail to check health and the restart count should remain 0.
	*/
	framework.ConformanceIt("should *not* be restarted with a exec \"cat /tmp/health\" liveness probe [NodeConformance]", func() {
		runLivenessTest(f, &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "liveness-exec",
				Labels: map[string]string{"test": "liveness"},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    "liveness",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
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

	/*
		Release : v1.9
		Testname: Pod liveness probe, using http endpoint, restart
		Description: A Pod is created with liveness probe on http endpoint /healthz. The http handler on the /healthz will return a http error after 10 seconds since the Pod is started. This MUST result in liveness check failure. The Pod MUST now be killed and restarted incrementing restart count to 1.
	*/
	framework.ConformanceIt("should be restarted with a /healthz http liveness probe [NodeConformance]", func() {
		runLivenessTest(f, &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "liveness-http",
				Labels: map[string]string{"test": "liveness"},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    "liveness",
						Image:   imageutils.GetE2EImage(imageutils.Liveness),
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

	/*
		Release : v1.9
		Testname: Pod liveness probe, using http endpoint, multiple restarts (slow)
		Description: A Pod is created with liveness probe on http endpoint /healthz. The http handler on the /healthz will return a http error after 10 seconds since the Pod is started. This MUST result in liveness check failure. The Pod MUST now be killed and restarted incrementing restart count to 1. The liveness probe must fail again after restart once the http handler for /healthz enpoind on the Pod returns an http error after 10 seconds from the start. Restart counts MUST increment everytime health check fails, measure upto 5 restart.
	*/
	framework.ConformanceIt("should have monotonically increasing restart count [NodeConformance]", func() {
		runLivenessTest(f, &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "liveness-http",
				Labels: map[string]string{"test": "liveness"},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    "liveness",
						Image:   imageutils.GetE2EImage(imageutils.Liveness),
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

	/*
		Release : v1.9
		Testname: Pod liveness probe, using http endpoint, failure
		Description: A Pod is created with liveness probe on http endpoint ‘/’. Liveness probe on this endpoint will not fail. When liveness probe does not fail then the restart count MUST remain zero.
	*/
	framework.ConformanceIt("should *not* be restarted with a /healthz http liveness probe [NodeConformance]", func() {
		runLivenessTest(f, &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "liveness-http",
				Labels: map[string]string{"test": "liveness"},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "liveness",
						Image: imageutils.GetE2EImage(imageutils.Nginx),
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
							FailureThreshold:    5, // to accommodate nodes which are slow in bringing up containers.
						},
					},
				},
			},
		}, 0, defaultObservationTimeout)
	})

	/*
		Release : v1.9
		Testname: Pod liveness probe, docker exec, restart
		Description: A Pod is created with liveness probe with a Exec action on the Pod. If the liveness probe call  does not return within the timeout specified, liveness probe MUST restart the Pod.
	*/
	It("should be restarted with a docker exec liveness probe with timeout ", func() {
		// TODO: enable this test once the default exec handler supports timeout.
		framework.Skipf("The default exec handler, dockertools.NativeExecHandler, does not support timeouts due to a limitation in the Docker Remote API")
		runLivenessTest(f, &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "liveness-exec",
				Labels: map[string]string{"test": "liveness"},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    "liveness",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
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

	/*
		Release : v1.14
		Testname: Pod http liveness probe, redirected to a local address
		Description: A Pod is created with liveness probe on http endpoint /redirect?loc=healthz. The http handler on the /redirect will redirect to the /healthz endpoint, which will return a http error after 10 seconds since the Pod is started. This MUST result in liveness check failure. The Pod MUST now be killed and restarted incrementing restart count to 1.
	*/
	It("should be restarted with a local redirect http liveness probe", func() {
		runLivenessTest(f, &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "liveness-http-redirect",
				Labels: map[string]string{"test": "liveness"},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    "liveness",
						Image:   imageutils.GetE2EImage(imageutils.Liveness),
						Command: []string{"/server"},
						LivenessProbe: &v1.Probe{
							Handler: v1.Handler{
								HTTPGet: &v1.HTTPGetAction{
									Path: "/redirect?loc=" + url.QueryEscape("/healthz"),
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

	/*
		Release : v1.14
		Testname: Pod http liveness probe, redirected to a non-local address
		Description: A Pod is created with liveness probe on http endpoint /redirect with a redirect to http://0.0.0.0/. The http handler on the /redirect should not follow the redirect, but instead treat it as a success and generate an event.
	*/
	It("should *not* be restarted with a non-local redirect http liveness probe", func() {
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "liveness-http-redirect",
				Labels: map[string]string{"test": "liveness"},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    "liveness",
						Image:   imageutils.GetE2EImage(imageutils.Liveness),
						Command: []string{"/server"},
						LivenessProbe: &v1.Probe{
							Handler: v1.Handler{
								HTTPGet: &v1.HTTPGetAction{
									Path: "/redirect?loc=" + url.QueryEscape("http://0.0.0.0/"),
									Port: intstr.FromInt(8080),
								},
							},
							InitialDelaySeconds: 15,
							FailureThreshold:    1,
						},
					},
				},
			},
		}
		runLivenessTest(f, pod, 0, defaultObservationTimeout)
		// Expect an event of type "ProbeWarning".
		expectedEvent := fields.Set{
			"involvedObject.kind":      "Pod",
			"involvedObject.name":      pod.Name,
			"involvedObject.namespace": f.Namespace.Name,
			"reason":                   events.ContainerProbeWarning,
		}.AsSelector().String()
		framework.ExpectNoError(framework.WaitTimeoutForPodEvent(
			f.ClientSet, pod.Name, f.Namespace.Name, expectedEvent, "0.0.0.0", framework.PodEventTimeout))
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
					Image:          imageutils.GetE2EImage(imageutils.TestWebserver),
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
	initialRestartCount := podutil.GetExistingContainerStatus(pod.Status.ContainerStatuses, containerName).RestartCount
	framework.Logf("Initial restart count of pod %s is %d", pod.Name, initialRestartCount)

	// Wait for the restart state to be as desired.
	deadline := time.Now().Add(timeout)
	lastRestartCount := initialRestartCount
	observedRestarts := int32(0)
	for start := time.Now(); time.Now().Before(deadline); time.Sleep(2 * time.Second) {
		pod, err = podClient.Get(pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, fmt.Sprintf("getting pod %s", pod.Name))
		restartCount := podutil.GetExistingContainerStatus(pod.Status.ContainerStatuses, containerName).RestartCount
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
