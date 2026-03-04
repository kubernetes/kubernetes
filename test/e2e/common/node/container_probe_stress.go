/*
Copyright 2026 The Kubernetes Authors.

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

// This file contains stress tests for container probes (HTTP, TCP, gRPC).
// These tests verify that probes do not cause spurious container restarts
// when many containers are probed at high frequency.
//
// Related issue: https://github.com/kubernetes/kubernetes/issues/115782
// Related bug: https://github.com/kubernetes/kubernetes/issues/89898
//
// Target path in kubernetes/kubernetes:
//   test/e2e/common/node/container_probe_stress.go

package node

import (
	"context"
	"fmt"
	"time"

	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

const (
	// stressContainerCount is the number of containers per pod for stress testing.
	// 10 containers each with 1s probe interval = 10 probes/second on the kubelet.
	stressContainerCount = 10

	// stressObservationPeriod is how long we observe after all containers are running.
	// 2 minutes = ~120 probe cycles per container, sufficient to catch timing issues.
	stressObservationPeriod = 2 * time.Minute

	// stressProbeBasePortHTTP is the starting port for HTTP/TCP containers.
	stressProbeBasePortHTTP = 8080

	// stressProbeBasePortGRPC is the starting port for gRPC containers.
	stressProbeBasePortGRPC = 5000
)

var _ = SIGDescribe("Probing container", framework.WithSerial(), func() {
	f := framework.NewDefaultFramework("container-probe-stress")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	/*
		Release: v1.32
		Testname: Probe stress, HTTP liveness, many containers
		Description: Create a pod with many containers, each with an HTTP
		liveness probe at 1-second intervals. After an observation period,
		verify that no container has restarted. This ensures the kubelet's
		probe worker pool handles high probe concurrency without causing
		spurious failures.
	*/
	f.It("should not restart containers when many HTTP liveness probes fire at 1s intervals", func(ctx context.Context) {
		pod := buildHTTPProbeStressPod(stressContainerCount)
		runProbeStressTest(ctx, f, pod)
	})

	/*
		Release: v1.32
		Testname: Probe stress, TCP liveness, many containers
		Description: Create a pod with many containers, each with a TCP
		liveness probe at 1-second intervals. After an observation period,
		verify that no container has restarted.
	*/
	f.It("should not restart containers when many TCP liveness probes fire at 1s intervals", func(ctx context.Context) {
		pod := buildTCPProbeStressPod(stressContainerCount)
		runProbeStressTest(ctx, f, pod)
	})

	/*
		Release: v1.32
		Testname: Probe stress, gRPC liveness, many containers
		Description: Create a pod with many containers, each with a gRPC
		liveness probe at 1-second intervals. After an observation period,
		verify that no container has restarted.
	*/
	f.It("should not restart containers when many gRPC liveness probes fire at 1s intervals", func(ctx context.Context) {
		pod := buildGRPCProbeStressPod(stressContainerCount)
		runProbeStressTest(ctx, f, pod)
	})
})

// runProbeStressTest creates the pod, waits for all containers to be running,
// observes for stressObservationPeriod, then asserts no container restarted.
func runProbeStressTest(ctx context.Context, f *framework.Framework, pod *v1.Pod) {
	podClient := e2epod.NewPodClient(f)

	pod = podClient.Create(ctx, pod)
	framework.Logf("Created stress test pod %s with %d containers", pod.Name, len(pod.Spec.Containers))

	// Wait for all containers to be running and ready
	err := e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod)
	framework.ExpectNoError(err, "waiting for stress test pod to be running")

	framework.Logf("All containers running. Observing for %v to detect spurious restarts...", stressObservationPeriod)
	time.Sleep(stressObservationPeriod)

	// Re-fetch pod to get current container statuses
	pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err, "getting pod after observation period")

	for _, cs := range pod.Status.ContainerStatuses {
		gomega.Expect(cs.RestartCount).To(
			gomega.BeZero(),
			fmt.Sprintf("container %q restarted %d times during observation period — "+
				"probe socket handling may be causing spurious failures under load",
				cs.Name, cs.RestartCount),
		)
	}
	framework.Logf("No containers restarted after %v — probe stress test passed", stressObservationPeriod)
}

// buildHTTPProbeStressPod creates a pod with N containers, each running
// agnhost netexec on a unique port with an HTTP liveness probe.
func buildHTTPProbeStressPod(containerCount int) *v1.Pod {
	containers := make([]v1.Container, containerCount)
	for i := 0; i < containerCount; i++ {
		port := stressProbeBasePortHTTP + i
		containers[i] = v1.Container{
			Name:  fmt.Sprintf("http-stress-%d", i),
			Image: imageutils.GetE2EImage(imageutils.Agnhost),
			Args:  []string{"netexec", fmt.Sprintf("--http-port=%d", port)},
			Ports: []v1.ContainerPort{{ContainerPort: int32(port)}},
			LivenessProbe: &v1.Probe{
				ProbeHandler: v1.ProbeHandler{
					HTTPGet: &v1.HTTPGetAction{
						Path: "/healthz",
						Port: intstr.FromInt32(int32(port)),
					},
				},
				PeriodSeconds:    1,
				TimeoutSeconds:   1,
				FailureThreshold: 3,
			},
		}
	}
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "probe-stress-http-" + string(uuid.NewUUID()),
		},
		Spec: v1.PodSpec{
			Containers: containers,
		},
	}
}

// buildTCPProbeStressPod creates a pod with N containers, each running
// agnhost netexec on a unique port with a TCP liveness probe.
func buildTCPProbeStressPod(containerCount int) *v1.Pod {
	containers := make([]v1.Container, containerCount)
	for i := 0; i < containerCount; i++ {
		port := stressProbeBasePortHTTP + i
		containers[i] = v1.Container{
			Name:  fmt.Sprintf("tcp-stress-%d", i),
			Image: imageutils.GetE2EImage(imageutils.Agnhost),
			Args:  []string{"netexec", fmt.Sprintf("--http-port=%d", port)},
			Ports: []v1.ContainerPort{{ContainerPort: int32(port)}},
			LivenessProbe: &v1.Probe{
				ProbeHandler: v1.ProbeHandler{
					TCPSocket: &v1.TCPSocketAction{
						Port: intstr.FromInt32(int32(port)),
					},
				},
				PeriodSeconds:    1,
				TimeoutSeconds:   1,
				FailureThreshold: 3,
			},
		}
	}
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "probe-stress-tcp-" + string(uuid.NewUUID()),
		},
		Spec: v1.PodSpec{
			Containers: containers,
		},
	}
}

// buildGRPCProbeStressPod creates a pod with N containers, each running
// agnhost grpc-health-checking on a unique port with a gRPC liveness probe.
func buildGRPCProbeStressPod(containerCount int) *v1.Pod {
	containers := make([]v1.Container, containerCount)
	for i := 0; i < containerCount; i++ {
		port := stressProbeBasePortGRPC + i
		containers[i] = v1.Container{
			Name:  fmt.Sprintf("grpc-stress-%d", i),
			Image: imageutils.GetE2EImage(imageutils.Agnhost),
			Args:  []string{"grpc-health-checking", fmt.Sprintf("--port=%d", port)},
			Ports: []v1.ContainerPort{{ContainerPort: int32(port)}},
			LivenessProbe: &v1.Probe{
				ProbeHandler: v1.ProbeHandler{
					GRPC: &v1.GRPCAction{
						Port: int32(port),
					},
				},
				PeriodSeconds:    1,
				TimeoutSeconds:   1,
				FailureThreshold: 3,
			},
		}
	}
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "probe-stress-grpc-" + string(uuid.NewUUID()),
		},
		Spec: v1.PodSpec{
			Containers: containers,
		},
	}
}
