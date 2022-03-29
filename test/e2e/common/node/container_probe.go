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

package node

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/url"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eevents "k8s.io/kubernetes/test/e2e/framework/events"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

const (
	probeTestInitialDelaySeconds = 15

	defaultObservationTimeout = time.Minute * 4
)

var _ = SIGDescribe("Probing container", func() {
	f := framework.NewDefaultFramework("container-probe")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelBaseline
	var podClient *framework.PodClient
	probe := webserverProbeBuilder{}

	ginkgo.BeforeEach(func() {
		podClient = f.PodClient()
	})

	/*
		Release: v1.9
		Testname: Pod readiness probe, with initial delay
		Description: Create a Pod that is configured with a initial delay set on the readiness probe. Check the Pod Start time to compare to the initial delay. The Pod MUST be ready only after the specified initial delay.
	*/
	framework.ConformanceIt("with readiness probe should not be ready before initial delay and never restart [NodeConformance]", func() {
		containerName := "test-webserver"
		p := podClient.Create(testWebServerPodSpec(probe.withInitialDelay().build(), nil, containerName, 80))
		e2epod.WaitTimeoutForPodReadyInNamespace(f.ClientSet, p.Name, f.Namespace.Name, framework.PodStartTimeout)

		p, err := podClient.Get(context.TODO(), p.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		isReady, err := testutils.PodRunningReady(p)
		framework.ExpectNoError(err)
		if !isReady {
			framework.Failf("pod %s/%s should be ready", f.Namespace.Name, p.Name)
		}

		// We assume the pod became ready when the container became ready. This
		// is true for a single container pod.
		readyTime, err := GetTransitionTimeForReadyCondition(p)
		framework.ExpectNoError(err)
		startedTime, err := GetContainerStartedTime(p, containerName)
		framework.ExpectNoError(err)

		framework.Logf("Container started at %v, pod became ready at %v", startedTime, readyTime)
		initialDelay := probeTestInitialDelaySeconds * time.Second
		if readyTime.Sub(startedTime) < initialDelay {
			framework.Failf("Pod became ready before it's %v initial delay", initialDelay)
		}

		restartCount := getRestartCount(p)
		framework.ExpectEqual(restartCount, 0, "pod should have a restart count of 0 but got %v", restartCount)
	})

	/*
		Release: v1.9
		Testname: Pod readiness probe, failure
		Description: Create a Pod with a readiness probe that fails consistently. When this Pod is created,
			then the Pod MUST never be ready, never be running and restart count MUST be zero.
	*/
	framework.ConformanceIt("with readiness probe that fails should never be ready and never restart [NodeConformance]", func() {
		p := podClient.Create(testWebServerPodSpec(probe.withFailing().build(), nil, "test-webserver", 80))
		gomega.Consistently(func() (bool, error) {
			p, err := podClient.Get(context.TODO(), p.Name, metav1.GetOptions{})
			if err != nil {
				return false, err
			}
			return podutil.IsPodReady(p), nil
		}, 1*time.Minute, 1*time.Second).ShouldNot(gomega.BeTrue(), "pod should not be ready")

		p, err := podClient.Get(context.TODO(), p.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)

		isReady, _ := testutils.PodRunningReady(p)
		if isReady {
			framework.Failf("pod %s/%s should be not ready", f.Namespace.Name, p.Name)
		}

		restartCount := getRestartCount(p)
		framework.ExpectEqual(restartCount, 0, "pod should have a restart count of 0 but got %v", restartCount)
	})

	/*
		Release: v1.9
		Testname: Pod liveness probe, using local file, restart
		Description: Create a Pod with liveness probe that uses ExecAction handler to cat /temp/health file. The Container deletes the file /temp/health after 10 second, triggering liveness probe to fail. The Pod MUST now be killed and restarted incrementing restart count to 1.
	*/
	framework.ConformanceIt("should be restarted with a exec \"cat /tmp/health\" liveness probe [NodeConformance]", func() {
		cmd := []string{"/bin/sh", "-c", "echo ok >/tmp/health; sleep 10; rm -rf /tmp/health; sleep 600"}
		livenessProbe := &v1.Probe{
			ProbeHandler:        execHandler([]string{"cat", "/tmp/health"}),
			InitialDelaySeconds: 15,
			TimeoutSeconds:      5, // default 1s can be pretty aggressive in CI environments with low resources
			FailureThreshold:    1,
		}
		pod := busyBoxPodSpec(nil, livenessProbe, cmd)
		RunLivenessTest(f, pod, 1, defaultObservationTimeout)
	})

	/*
		Release: v1.9
		Testname: Pod liveness probe, using local file, no restart
		Description:  Pod is created with liveness probe that uses 'exec' command to cat /temp/health file. Liveness probe MUST not fail to check health and the restart count should remain 0.
	*/
	framework.ConformanceIt("should *not* be restarted with a exec \"cat /tmp/health\" liveness probe [NodeConformance]", func() {
		cmd := []string{"/bin/sh", "-c", "echo ok >/tmp/health; sleep 600"}
		livenessProbe := &v1.Probe{
			ProbeHandler:        execHandler([]string{"cat", "/tmp/health"}),
			InitialDelaySeconds: 15,
			TimeoutSeconds:      5, // default 1s can be pretty aggressive in CI environments with low resources
			FailureThreshold:    1,
		}
		pod := busyBoxPodSpec(nil, livenessProbe, cmd)
		RunLivenessTest(f, pod, 0, defaultObservationTimeout)
	})

	/*
		Release: v1.9
		Testname: Pod liveness probe, using http endpoint, restart
		Description: A Pod is created with liveness probe on http endpoint /healthz. The http handler on the /healthz will return a http error after 10 seconds since the Pod is started. This MUST result in liveness check failure. The Pod MUST now be killed and restarted incrementing restart count to 1.
	*/
	framework.ConformanceIt("should be restarted with a /healthz http liveness probe [NodeConformance]", func() {
		livenessProbe := &v1.Probe{
			ProbeHandler:        httpGetHandler("/healthz", 8080),
			InitialDelaySeconds: 15,
			FailureThreshold:    1,
		}
		pod := livenessPodSpec(f.Namespace.Name, nil, livenessProbe)
		RunLivenessTest(f, pod, 1, defaultObservationTimeout)
	})

	/*
		Release: v1.18
		Testname: Pod liveness probe, using tcp socket, no restart
		Description: A Pod is created with liveness probe on tcp socket 8080. The http handler on port 8080 will return http errors after 10 seconds, but the socket will remain open. Liveness probe MUST not fail to check health and the restart count should remain 0.
	*/
	framework.ConformanceIt("should *not* be restarted with a tcp:8080 liveness probe [NodeConformance]", func() {
		livenessProbe := &v1.Probe{
			ProbeHandler:        tcpSocketHandler(8080),
			InitialDelaySeconds: 15,
			FailureThreshold:    1,
		}
		pod := livenessPodSpec(f.Namespace.Name, nil, livenessProbe)
		RunLivenessTest(f, pod, 0, defaultObservationTimeout)
	})

	/*
		Release: v1.9
		Testname: Pod liveness probe, using http endpoint, multiple restarts (slow)
		Description: A Pod is created with liveness probe on http endpoint /healthz. The http handler on the /healthz will return a http error after 10 seconds since the Pod is started. This MUST result in liveness check failure. The Pod MUST now be killed and restarted incrementing restart count to 1. The liveness probe must fail again after restart once the http handler for /healthz enpoind on the Pod returns an http error after 10 seconds from the start. Restart counts MUST increment everytime health check fails, measure upto 5 restart.
	*/
	framework.ConformanceIt("should have monotonically increasing restart count [NodeConformance]", func() {
		livenessProbe := &v1.Probe{
			ProbeHandler:        httpGetHandler("/healthz", 8080),
			InitialDelaySeconds: 5,
			FailureThreshold:    1,
		}
		pod := livenessPodSpec(f.Namespace.Name, nil, livenessProbe)
		// ~2 minutes backoff timeouts + 4 minutes defaultObservationTimeout + 2 minutes for each pod restart
		RunLivenessTest(f, pod, 5, 2*time.Minute+defaultObservationTimeout+4*2*time.Minute)
	})

	/*
		Release: v1.9
		Testname: Pod liveness probe, using http endpoint, failure
		Description: A Pod is created with liveness probe on http endpoint '/'. Liveness probe on this endpoint will not fail. When liveness probe does not fail then the restart count MUST remain zero.
	*/
	framework.ConformanceIt("should *not* be restarted with a /healthz http liveness probe [NodeConformance]", func() {
		livenessProbe := &v1.Probe{
			ProbeHandler:        httpGetHandler("/", 80),
			InitialDelaySeconds: 15,
			TimeoutSeconds:      5,
			FailureThreshold:    5, // to accommodate nodes which are slow in bringing up containers.
		}
		pod := testWebServerPodSpec(nil, livenessProbe, "test-webserver", 80)
		RunLivenessTest(f, pod, 0, defaultObservationTimeout)
	})

	/*
		Release: v1.9
		Testname: Pod liveness probe, container exec timeout, restart
		Description: A Pod is created with liveness probe with a Exec action on the Pod. If the liveness probe call does not return within the timeout specified, liveness probe MUST restart the Pod.
	*/
	ginkgo.It("should be restarted with an exec liveness probe with timeout [MinimumKubeletVersion:1.20] [NodeConformance]", func() {
		cmd := []string{"/bin/sh", "-c", "sleep 600"}
		livenessProbe := &v1.Probe{
			ProbeHandler:        execHandler([]string{"/bin/sh", "-c", "sleep 10"}),
			InitialDelaySeconds: 15,
			TimeoutSeconds:      1,
			FailureThreshold:    1,
		}
		pod := busyBoxPodSpec(nil, livenessProbe, cmd)
		RunLivenessTest(f, pod, 1, defaultObservationTimeout)
	})

	/*
		Release: v1.20
		Testname: Pod readiness probe, container exec timeout, not ready
		Description: A Pod is created with readiness probe with a Exec action on the Pod. If the readiness probe call does not return within the timeout specified, readiness probe MUST not be Ready.
	*/
	ginkgo.It("should not be ready with an exec readiness probe timeout [MinimumKubeletVersion:1.20] [NodeConformance]", func() {
		cmd := []string{"/bin/sh", "-c", "sleep 600"}
		readinessProbe := &v1.Probe{
			ProbeHandler:        execHandler([]string{"/bin/sh", "-c", "sleep 10"}),
			InitialDelaySeconds: 15,
			TimeoutSeconds:      1,
			FailureThreshold:    1,
		}
		pod := busyBoxPodSpec(readinessProbe, nil, cmd)
		runReadinessFailTest(f, pod, time.Minute)
	})

	/*
		Release: v1.21
		Testname: Pod liveness probe, container exec timeout, restart
		Description: A Pod is created with liveness probe with a Exec action on the Pod. If the liveness probe call does not return within the timeout specified, liveness probe MUST restart the Pod. When ExecProbeTimeout feature gate is disabled and cluster is using dockershim, the timeout is ignored BUT a failing liveness probe MUST restart the Pod.
	*/
	ginkgo.It("should be restarted with a failing exec liveness probe that took longer than the timeout", func() {
		cmd := []string{"/bin/sh", "-c", "sleep 600"}
		livenessProbe := &v1.Probe{
			ProbeHandler:        execHandler([]string{"/bin/sh", "-c", "sleep 10 & exit 1"}),
			InitialDelaySeconds: 15,
			TimeoutSeconds:      1,
			FailureThreshold:    1,
		}
		pod := busyBoxPodSpec(nil, livenessProbe, cmd)
		RunLivenessTest(f, pod, 1, defaultObservationTimeout)
	})

	/*
		Release: v1.14
		Testname: Pod http liveness probe, redirected to a local address
		Description: A Pod is created with liveness probe on http endpoint /redirect?loc=healthz. The http handler on the /redirect will redirect to the /healthz endpoint, which will return a http error after 10 seconds since the Pod is started. This MUST result in liveness check failure. The Pod MUST now be killed and restarted incrementing restart count to 1.
	*/
	ginkgo.It("should be restarted with a local redirect http liveness probe", func() {
		livenessProbe := &v1.Probe{
			ProbeHandler:        httpGetHandler("/redirect?loc="+url.QueryEscape("/healthz"), 8080),
			InitialDelaySeconds: 15,
			FailureThreshold:    1,
		}
		pod := livenessPodSpec(f.Namespace.Name, nil, livenessProbe)
		RunLivenessTest(f, pod, 1, defaultObservationTimeout)
	})

	/*
		Release: v1.14
		Testname: Pod http liveness probe, redirected to a non-local address
		Description: A Pod is created with liveness probe on http endpoint /redirect with a redirect to http://0.0.0.0/. The http handler on the /redirect should not follow the redirect, but instead treat it as a success and generate an event.
	*/
	ginkgo.It("should *not* be restarted with a non-local redirect http liveness probe", func() {
		livenessProbe := &v1.Probe{
			ProbeHandler:        httpGetHandler("/redirect?loc="+url.QueryEscape("http://0.0.0.0/"), 8080),
			InitialDelaySeconds: 15,
			FailureThreshold:    1,
		}
		pod := livenessPodSpec(f.Namespace.Name, nil, livenessProbe)
		RunLivenessTest(f, pod, 0, defaultObservationTimeout)
		// Expect an event of type "ProbeWarning".
		expectedEvent := fields.Set{
			"involvedObject.kind":      "Pod",
			"involvedObject.name":      pod.Name,
			"involvedObject.namespace": f.Namespace.Name,
			"reason":                   events.ContainerProbeWarning,
		}.AsSelector().String()
		framework.ExpectNoError(e2eevents.WaitTimeoutForEvent(
			f.ClientSet, f.Namespace.Name, expectedEvent, "Probe terminated redirects, Response body: <a href=\"http://0.0.0.0/\">Found</a>.", framework.PodEventTimeout))
	})

	/*
		Release: v1.16
		Testname: Pod startup probe restart
		Description: A Pod is created with a failing startup probe. The Pod MUST be killed and restarted incrementing restart count to 1, even if liveness would succeed.
	*/
	ginkgo.It("should be restarted startup probe fails", func() {
		cmd := []string{"/bin/sh", "-c", "sleep 600"}
		livenessProbe := &v1.Probe{
			ProbeHandler: v1.ProbeHandler{
				Exec: &v1.ExecAction{
					Command: []string{"/bin/true"},
				},
			},
			InitialDelaySeconds: 15,
			FailureThreshold:    1,
		}
		startupProbe := &v1.Probe{
			ProbeHandler: v1.ProbeHandler{
				Exec: &v1.ExecAction{
					Command: []string{"/bin/false"},
				},
			},
			InitialDelaySeconds: 15,
			FailureThreshold:    3,
		}
		pod := startupPodSpec(startupProbe, nil, livenessProbe, cmd)
		RunLivenessTest(f, pod, 1, defaultObservationTimeout)
	})

	/*
		Release: v1.16
		Testname: Pod liveness probe delayed (long) by startup probe
		Description: A Pod is created with failing liveness and startup probes. Liveness probe MUST NOT fail until startup probe expires.
	*/
	ginkgo.It("should *not* be restarted by liveness probe because startup probe delays it", func() {
		cmd := []string{"/bin/sh", "-c", "sleep 600"}
		livenessProbe := &v1.Probe{
			ProbeHandler: v1.ProbeHandler{
				Exec: &v1.ExecAction{
					Command: []string{"/bin/false"},
				},
			},
			InitialDelaySeconds: 15,
			FailureThreshold:    1,
		}
		startupProbe := &v1.Probe{
			ProbeHandler: v1.ProbeHandler{
				Exec: &v1.ExecAction{
					Command: []string{"/bin/false"},
				},
			},
			InitialDelaySeconds: 15,
			FailureThreshold:    60,
		}
		pod := startupPodSpec(startupProbe, nil, livenessProbe, cmd)
		RunLivenessTest(f, pod, 0, defaultObservationTimeout)
	})

	/*
		Release: v1.16
		Testname: Pod liveness probe fails after startup success
		Description: A Pod is created with failing liveness probe and delayed startup probe that uses 'exec' command to cat /temp/health file. The Container is started by creating /tmp/startup after 10 seconds, triggering liveness probe to fail. The Pod MUST now be killed and restarted incrementing restart count to 1.
	*/
	ginkgo.It("should be restarted by liveness probe after startup probe enables it", func() {
		cmd := []string{"/bin/sh", "-c", "sleep 10; echo ok >/tmp/startup; sleep 600"}
		livenessProbe := &v1.Probe{
			ProbeHandler: v1.ProbeHandler{
				Exec: &v1.ExecAction{
					Command: []string{"/bin/false"},
				},
			},
			InitialDelaySeconds: 15,
			FailureThreshold:    1,
		}
		startupProbe := &v1.Probe{
			ProbeHandler: v1.ProbeHandler{
				Exec: &v1.ExecAction{
					Command: []string{"cat", "/tmp/startup"},
				},
			},
			InitialDelaySeconds: 15,
			FailureThreshold:    60,
		}
		pod := startupPodSpec(startupProbe, nil, livenessProbe, cmd)
		RunLivenessTest(f, pod, 1, defaultObservationTimeout)
	})

	/*
		Release: v1.16
		Testname: Pod readiness probe, delayed by startup probe
		Description: A Pod is created with startup and readiness probes. The Container is started by creating /tmp/startup after 45 seconds, delaying the ready state by this amount of time. This is similar to the "Pod readiness probe, with initial delay" test.
	*/
	ginkgo.It("should be ready immediately after startupProbe succeeds", func() {
		// Probe workers sleep at Kubelet start for a random time which is at most PeriodSeconds
		// this test requires both readiness and startup workers running before updating statuses
		// to avoid flakes, ensure sleep before startup (32s) > readinessProbe.PeriodSeconds
		cmd := []string{"/bin/sh", "-c", "echo ok >/tmp/health; sleep 32; echo ok >/tmp/startup; sleep 600"}
		readinessProbe := &v1.Probe{
			ProbeHandler:        execHandler([]string{"/bin/cat", "/tmp/health"}),
			InitialDelaySeconds: 0,
			PeriodSeconds:       30,
		}
		startupProbe := &v1.Probe{
			ProbeHandler:        execHandler([]string{"/bin/cat", "/tmp/startup"}),
			InitialDelaySeconds: 0,
			FailureThreshold:    120,
			PeriodSeconds:       5,
		}
		p := podClient.Create(startupPodSpec(startupProbe, readinessProbe, nil, cmd))

		p, err := podClient.Get(context.TODO(), p.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)

		err = e2epod.WaitForPodContainerStarted(f.ClientSet, f.Namespace.Name, p.Name, 0, framework.PodStartTimeout)
		framework.ExpectNoError(err)
		startedTime := time.Now()

		// We assume the pod became ready when the container became ready. This
		// is true for a single container pod.
		err = e2epod.WaitTimeoutForPodReadyInNamespace(f.ClientSet, p.Name, f.Namespace.Name, framework.PodStartTimeout)
		framework.ExpectNoError(err)
		readyTime := time.Now()

		p, err = podClient.Get(context.TODO(), p.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)

		isReady, err := testutils.PodRunningReady(p)
		framework.ExpectNoError(err)
		if !isReady {
			framework.Failf("pod %s/%s should be ready", f.Namespace.Name, p.Name)
		}

		readyIn := readyTime.Sub(startedTime)
		framework.Logf("Container started at %v, pod became ready at %v, %v after startupProbe succeeded", startedTime, readyTime, readyIn)
		if readyIn < 0 {
			framework.Failf("Pod became ready before startupProbe succeeded")
		}
		if readyIn > 25*time.Second {
			framework.Failf("Pod became ready in %v, more than 25s after startupProbe succeeded. It means that the delay readiness probes were not initiated immediately after startup finished.", readyIn)
		}
	})

	/*
		Release: v1.21
		Testname: Set terminationGracePeriodSeconds for livenessProbe
		Description: A pod with a long terminationGracePeriod is created with a shorter livenessProbe-level terminationGracePeriodSeconds. We confirm the shorter termination period is used.
	*/
	ginkgo.It("should override timeoutGracePeriodSeconds when LivenessProbe field is set [Feature:ProbeTerminationGracePeriod]", func() {
		pod := e2epod.NewAgnhostPod(f.Namespace.Name, "liveness-override-"+string(uuid.NewUUID()), nil, nil, nil, "/bin/sh", "-c", "sleep 1000")
		longGracePeriod := int64(500)
		pod.Spec.TerminationGracePeriodSeconds = &longGracePeriod

		// probe will fail since pod has no http endpoints
		shortGracePeriod := int64(5)
		pod.Spec.Containers[0].LivenessProbe = &v1.Probe{
			ProbeHandler: v1.ProbeHandler{
				HTTPGet: &v1.HTTPGetAction{
					Path: "/healthz",
					Port: intstr.FromInt(8080),
				},
			},
			InitialDelaySeconds:           10,
			FailureThreshold:              1,
			TerminationGracePeriodSeconds: &shortGracePeriod,
		}

		// 10s delay + 10s period + 5s grace period = 25s < 30s << pod-level timeout 500
		RunLivenessTest(f, pod, 1, time.Second*30)
	})

	/*
		Release: v1.21
		Testname: Set terminationGracePeriodSeconds for startupProbe
		Description: A pod with a long terminationGracePeriod is created with a shorter startupProbe-level terminationGracePeriodSeconds. We confirm the shorter termination period is used.
	*/
	ginkgo.It("should override timeoutGracePeriodSeconds when StartupProbe field is set [Feature:ProbeTerminationGracePeriod]", func() {
		pod := e2epod.NewAgnhostPod(f.Namespace.Name, "startup-override-"+string(uuid.NewUUID()), nil, nil, nil, "/bin/sh", "-c", "sleep 1000")
		longGracePeriod := int64(500)
		pod.Spec.TerminationGracePeriodSeconds = &longGracePeriod

		// startup probe will fail since pod will sleep for 1000s before becoming ready
		shortGracePeriod := int64(5)
		pod.Spec.Containers[0].StartupProbe = &v1.Probe{
			ProbeHandler:                  execHandler([]string{"/bin/cat", "/tmp/startup"}),
			InitialDelaySeconds:           10,
			FailureThreshold:              1,
			TerminationGracePeriodSeconds: &shortGracePeriod,
		}
		// liveness probe always succeeds
		pod.Spec.Containers[0].LivenessProbe = &v1.Probe{
			ProbeHandler: v1.ProbeHandler{
				Exec: &v1.ExecAction{
					Command: []string{"/bin/true"},
				},
			},
			InitialDelaySeconds: 15,
			FailureThreshold:    1,
		}

		// 10s delay + 10s period + 5s grace period = 25s < 30s << pod-level timeout 500
		RunLivenessTest(f, pod, 1, time.Second*30)
	})

	/*
		Release: v1.23
		Testname: Pod liveness probe, using grpc call, success
		Description: A Pod is created with liveness probe on grpc service. Liveness probe on this endpoint will not fail. When liveness probe does not fail then the restart count MUST remain zero.
	*/
	ginkgo.It("should *not* be restarted with a GRPC liveness probe [NodeConformance]", func() {
		livenessProbe := &v1.Probe{
			ProbeHandler: v1.ProbeHandler{
				GRPC: &v1.GRPCAction{
					Port:    2379,
					Service: nil,
				},
			},
			InitialDelaySeconds: probeTestInitialDelaySeconds,
			TimeoutSeconds:      5, // default 1s can be pretty aggressive in CI environments with low resources
			FailureThreshold:    1,
		}

		pod := gRPCServerPodSpec(nil, livenessProbe, "etcd")
		RunLivenessTest(f, pod, 0, defaultObservationTimeout)
	})

	/*
			Release: v1.23
			Testname: Pod liveness probe, using grpc call, failure
			Description: A Pod is created with liveness probe on grpc service. Liveness probe on this endpoint should fail because of wrong probe port.
		                 When liveness probe does  fail then the restart count should +1.
	*/
	ginkgo.It("should be restarted with a GRPC liveness probe [NodeConformance]", func() {
		livenessProbe := &v1.Probe{
			ProbeHandler: v1.ProbeHandler{
				GRPC: &v1.GRPCAction{
					Port: 2333, // this port is wrong
				},
			},
			InitialDelaySeconds: probeTestInitialDelaySeconds * 4,
			TimeoutSeconds:      5, // default 1s can be pretty aggressive in CI environments with low resources
			FailureThreshold:    1,
		}
		pod := gRPCServerPodSpec(nil, livenessProbe, "etcd")
		RunLivenessTest(f, pod, 1, defaultObservationTimeout)
	})

	ginkgo.It("should mark readiness on pods to false while pod is in progress of terminating when a pod has a readiness probe", func() {
		podName := "probe-test-" + string(uuid.NewUUID())
		podClient := f.PodClient()
		terminationGracePeriod := int64(30)
		script := `
_term() {
	rm -f /tmp/ready
	sleep 30
	exit 0
}
trap _term SIGTERM

touch /tmp/ready

while true; do
  echo \"hello\"
  sleep 10
done
			`

		// Create Pod
		podClient.Create(&v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: podName,
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Image:   imageutils.GetE2EImage(imageutils.Agnhost),
						Name:    podName,
						Command: []string{"/bin/bash"},
						Args:    []string{"-c", script},
						ReadinessProbe: &v1.Probe{
							ProbeHandler: v1.ProbeHandler{
								Exec: &v1.ExecAction{
									Command: []string{"cat", "/tmp/ready"},
								},
							},
							FailureThreshold:    1,
							InitialDelaySeconds: 5,
							PeriodSeconds:       2,
						},
					},
				},
				TerminationGracePeriodSeconds: &terminationGracePeriod,
			},
		})

		// verify pods are running and ready
		err := e2epod.WaitForPodsRunningReady(f.ClientSet, f.Namespace.Name, 1, 0, f.Timeouts.PodStart, map[string]string{})
		framework.ExpectNoError(err)

		// Shutdown pod. Readiness should change to false
		podClient.Delete(context.Background(), podName, metav1.DeleteOptions{})
		err = wait.PollImmediate(framework.Poll, f.Timeouts.PodDelete, func() (bool, error) {
			pod, err := podClient.Get(context.Background(), podName, metav1.GetOptions{})
			if err != nil {
				return false, err
			}
			// verify the pod ready status has reported not ready
			return podutil.IsPodReady(pod) == false, nil
		})
		framework.ExpectNoError(err)
	})

	ginkgo.It("should mark readiness on pods to false and disable liveness probes while pod is in progress of terminating", func() {
		podName := "probe-test-" + string(uuid.NewUUID())
		podClient := f.PodClient()
		terminationGracePeriod := int64(30)
		script := `
_term() { 
	rm -f /tmp/ready
	rm -f /tmp/liveness
	sleep 20
	exit 0
}
trap _term SIGTERM

touch /tmp/ready
touch /tmp/liveness

while true; do 
  echo \"hello\"
  sleep 10
done
`

		// Create Pod
		podClient.Create(&v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: podName,
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Image:   imageutils.GetE2EImage(imageutils.Agnhost),
						Name:    podName,
						Command: []string{"/bin/bash"},
						Args:    []string{"-c", script},
						ReadinessProbe: &v1.Probe{
							ProbeHandler: v1.ProbeHandler{
								Exec: &v1.ExecAction{
									Command: []string{"cat", "/tmp/ready"},
								},
							},
							FailureThreshold: 1,
							// delay startup to make sure the script script has
							// time to create the ready+liveness files
							InitialDelaySeconds: 5,
							PeriodSeconds:       2,
						},
						LivenessProbe: &v1.Probe{
							ProbeHandler: v1.ProbeHandler{
								Exec: &v1.ExecAction{
									Command: []string{"cat", "/tmp/liveness"},
								},
							},
							FailureThreshold: 1,
							// delay startup to make sure the script script has
							// time to create the ready+liveness files
							InitialDelaySeconds: 5,
							PeriodSeconds:       1,
						},
					},
				},
				TerminationGracePeriodSeconds: &terminationGracePeriod,
			},
		})

		// verify pods are running and ready
		err := e2epod.WaitForPodsRunningReady(f.ClientSet, f.Namespace.Name, 1, 0, f.Timeouts.PodStart, map[string]string{})
		framework.ExpectNoError(err)

		// Shutdown pod. Readiness should change to false
		podClient.Delete(context.Background(), podName, metav1.DeleteOptions{})

		// Wait for pod to go unready
		err = wait.PollImmediate(framework.Poll, f.Timeouts.PodDelete, func() (bool, error) {
			pod, err := podClient.Get(context.Background(), podName, metav1.GetOptions{})
			if err != nil {
				return false, err
			}
			// verify the pod ready status has reported not ready
			return podutil.IsPodReady(pod) == false, nil
		})
		framework.ExpectNoError(err)

		// Verify there are zero liveness failures since they are turned off
		// during pod termination
		gomega.Consistently(func() (bool, error) {
			items, err := f.ClientSet.CoreV1().Events(f.Namespace.Name).List(context.Background(), metav1.ListOptions{})
			framework.ExpectNoError(err)
			for _, event := range items.Items {
				// Search only for the pod we are interested in
				if event.InvolvedObject.Name != podName {
					continue
				}
				if strings.Contains(event.Message, "failed liveness probe") {
					return true, errors.New("should not see liveness probe failures")
				}
			}
			return false, nil
		}, 1*time.Minute, framework.Poll).ShouldNot(gomega.BeTrue(), "should not see liveness probes")
	})
})

// GetContainerStartedTime returns the time when the given container started and error if any
func GetContainerStartedTime(p *v1.Pod, containerName string) (time.Time, error) {
	for _, status := range p.Status.ContainerStatuses {
		if status.Name != containerName {
			continue
		}
		if status.State.Running == nil {
			return time.Time{}, fmt.Errorf("container is not running")
		}
		return status.State.Running.StartedAt.Time, nil
	}
	return time.Time{}, fmt.Errorf("cannot find container named %q", containerName)
}

// GetTransitionTimeForReadyCondition returns the time when the given pod became ready and error if any
func GetTransitionTimeForReadyCondition(p *v1.Pod) (time.Time, error) {
	for _, cond := range p.Status.Conditions {
		if cond.Type == v1.PodReady {
			return cond.LastTransitionTime.Time, nil
		}
	}
	return time.Time{}, fmt.Errorf("no ready condition can be found for pod")
}

func getRestartCount(p *v1.Pod) int {
	count := 0
	for _, containerStatus := range p.Status.ContainerStatuses {
		count += int(containerStatus.RestartCount)
	}
	return count
}

func testWebServerPodSpec(readinessProbe, livenessProbe *v1.Probe, containerName string, port int) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "test-webserver-" + string(uuid.NewUUID())},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:           containerName,
					Image:          imageutils.GetE2EImage(imageutils.Agnhost),
					Args:           []string{"test-webserver"},
					Ports:          []v1.ContainerPort{{ContainerPort: int32(port)}},
					LivenessProbe:  livenessProbe,
					ReadinessProbe: readinessProbe,
				},
			},
		},
	}
}

func busyBoxPodSpec(readinessProbe, livenessProbe *v1.Probe, cmd []string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "busybox-" + string(uuid.NewUUID()),
			Labels: map[string]string{"test": "liveness"},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:           "busybox",
					Image:          imageutils.GetE2EImage(imageutils.BusyBox),
					Command:        cmd,
					LivenessProbe:  livenessProbe,
					ReadinessProbe: readinessProbe,
				},
			},
		},
	}
}

func livenessPodSpec(namespace string, readinessProbe, livenessProbe *v1.Probe) *v1.Pod {
	pod := e2epod.NewAgnhostPod(namespace, "liveness-"+string(uuid.NewUUID()), nil, nil, nil, "liveness")
	pod.ObjectMeta.Labels = map[string]string{"test": "liveness"}
	pod.Spec.Containers[0].LivenessProbe = livenessProbe
	pod.Spec.Containers[0].ReadinessProbe = readinessProbe
	return pod
}

func startupPodSpec(startupProbe, readinessProbe, livenessProbe *v1.Probe, cmd []string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "startup-" + string(uuid.NewUUID()),
			Labels: map[string]string{"test": "startup"},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:           "busybox",
					Image:          imageutils.GetE2EImage(imageutils.BusyBox),
					Command:        cmd,
					LivenessProbe:  livenessProbe,
					ReadinessProbe: readinessProbe,
					StartupProbe:   startupProbe,
				},
			},
		},
	}
}

func execHandler(cmd []string) v1.ProbeHandler {
	return v1.ProbeHandler{
		Exec: &v1.ExecAction{
			Command: cmd,
		},
	}
}

func httpGetHandler(path string, port int) v1.ProbeHandler {
	return v1.ProbeHandler{
		HTTPGet: &v1.HTTPGetAction{
			Path: path,
			Port: intstr.FromInt(port),
		},
	}
}

func tcpSocketHandler(port int) v1.ProbeHandler {
	return v1.ProbeHandler{
		TCPSocket: &v1.TCPSocketAction{
			Port: intstr.FromInt(port),
		},
	}
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
		ProbeHandler: httpGetHandler("/", 80),
	}
	if b.initialDelay {
		probe.InitialDelaySeconds = probeTestInitialDelaySeconds
	}
	if b.failing {
		probe.HTTPGet.Port = intstr.FromInt(81)
	}
	return probe
}

// RunLivenessTest verifies the number of restarts for pod with given expected number of restarts
func RunLivenessTest(f *framework.Framework, pod *v1.Pod, expectNumRestarts int, timeout time.Duration) {
	podClient := f.PodClient()
	ns := f.Namespace.Name
	gomega.Expect(pod.Spec.Containers).NotTo(gomega.BeEmpty())
	containerName := pod.Spec.Containers[0].Name
	// At the end of the test, clean up by removing the pod.
	defer func() {
		ginkgo.By("deleting the pod")
		podClient.Delete(context.TODO(), pod.Name, *metav1.NewDeleteOptions(0))
	}()
	ginkgo.By(fmt.Sprintf("Creating pod %s in namespace %s", pod.Name, ns))
	podClient.Create(pod)

	// Wait until the pod is not pending. (Here we need to check for something other than
	// 'Pending' other than checking for 'Running', since when failures occur, we go to
	// 'Terminated' which can cause indefinite blocking.)
	framework.ExpectNoError(e2epod.WaitForPodNotPending(f.ClientSet, ns, pod.Name),
		fmt.Sprintf("starting pod %s in namespace %s", pod.Name, ns))
	framework.Logf("Started pod %s in namespace %s", pod.Name, ns)

	// Check the pod's current state and verify that restartCount is present.
	ginkgo.By("checking the pod's current state and verifying that restartCount is present")
	pod, err := podClient.Get(context.TODO(), pod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err, fmt.Sprintf("getting pod %s in namespace %s", pod.Name, ns))
	initialRestartCount := podutil.GetExistingContainerStatus(pod.Status.ContainerStatuses, containerName).RestartCount
	framework.Logf("Initial restart count of pod %s is %d", pod.Name, initialRestartCount)

	// Wait for the restart state to be as desired.
	deadline := time.Now().Add(timeout)
	lastRestartCount := initialRestartCount
	observedRestarts := int32(0)
	for start := time.Now(); time.Now().Before(deadline); time.Sleep(2 * time.Second) {
		pod, err = podClient.Get(context.TODO(), pod.Name, metav1.GetOptions{})
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

func runReadinessFailTest(f *framework.Framework, pod *v1.Pod, notReadyUntil time.Duration) {
	podClient := f.PodClient()
	ns := f.Namespace.Name
	gomega.Expect(pod.Spec.Containers).NotTo(gomega.BeEmpty())

	// At the end of the test, clean up by removing the pod.
	defer func() {
		ginkgo.By("deleting the pod")
		podClient.Delete(context.TODO(), pod.Name, *metav1.NewDeleteOptions(0))
	}()
	ginkgo.By(fmt.Sprintf("Creating pod %s in namespace %s", pod.Name, ns))
	podClient.Create(pod)

	// Wait until the pod is not pending. (Here we need to check for something other than
	// 'Pending', since when failures occur, we go to 'Terminated' which can cause indefinite blocking.)
	framework.ExpectNoError(e2epod.WaitForPodNotPending(f.ClientSet, ns, pod.Name),
		fmt.Sprintf("starting pod %s in namespace %s", pod.Name, ns))
	framework.Logf("Started pod %s in namespace %s", pod.Name, ns)

	// Wait for the not ready state to be true for notReadyUntil duration
	deadline := time.Now().Add(notReadyUntil)
	for start := time.Now(); time.Now().Before(deadline); time.Sleep(2 * time.Second) {
		// poll for Not Ready
		if podutil.IsPodReady(pod) {
			framework.Failf("pod %s/%s - expected to be not ready", ns, pod.Name)
		}

		framework.Logf("pod %s/%s is not ready (%v elapsed)",
			ns, pod.Name, time.Since(start))
	}
}

func gRPCServerPodSpec(readinessProbe, livenessProbe *v1.Probe, containerName string) *v1.Pod {
	etcdLocalhostAddress := "127.0.0.1"
	if framework.TestContext.ClusterIsIPv6() {
		etcdLocalhostAddress = "::1"
	}
	etcdURL := fmt.Sprintf("http://%s", net.JoinHostPort(etcdLocalhostAddress, "2379"))
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "test-grpc-" + string(uuid.NewUUID())},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  containerName,
					Image: imageutils.GetE2EImage(imageutils.Etcd),
					Command: []string{
						"/usr/local/bin/etcd",
						"--listen-client-urls",
						"http://0.0.0.0:2379", //should listen on all addresses
						"--advertise-client-urls",
						etcdURL,
					},
					// 2380 is an automatic peer URL
					Ports:          []v1.ContainerPort{{ContainerPort: int32(2379)}, {ContainerPort: int32(2380)}},
					LivenessProbe:  livenessProbe,
					ReadinessProbe: readinessProbe,
				},
			},
		},
	}
}
