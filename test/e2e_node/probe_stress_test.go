//go:build linux

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

package e2enode

import (
	"context"
	"fmt"
	"time"

	"github.com/onsi/ginkgo/v2"
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
	probeStressNumContainers = 50
	probeStressPeriodSeconds = 1
	probeStressWaitTime      = 2 * time.Minute
)

var _ = SIGDescribe("Probe Stress", framework.WithSerial(), func() {
	f := framework.NewDefaultFramework("probe-stress")
	addAfterEachForCleaningUpPods(f)
	// LevelPrivileged is required because the stress tests create pods with many containers
	// that may require elevated permissions for networking and resource allocation
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.Context("HTTP liveness probes", func() {
		ginkgo.It("should not cause unexpected container restarts under load", func(ctx context.Context) {
			pod := createPodWithHTTPProbes(probeStressNumContainers)
			runProbeStressTest(ctx, f, pod)
		})
	})

	ginkgo.Context("TCP liveness probes", func() {
		ginkgo.It("should not cause unexpected container restarts under load", func(ctx context.Context) {
			pod := createPodWithTCPProbes(probeStressNumContainers)
			runProbeStressTest(ctx, f, pod)
		})
	})

	ginkgo.Context("gRPC liveness probes", func() {
		ginkgo.It("should not cause unexpected container restarts under load", func(ctx context.Context) {
			pod := createPodWithGRPCProbes(probeStressNumContainers)
			runProbeStressTest(ctx, f, pod)
		})
	})
})

// runProbeStressTest creates a pod with many containers, waits for them to be running,
// and verifies that all containers keep running without restarting unexpectedly.
func runProbeStressTest(ctx context.Context, f *framework.Framework, pod *v1.Pod) {
	ginkgo.By(fmt.Sprintf("Creating pod %s with %d containers", pod.Name, len(pod.Spec.Containers)))
	pod = e2epod.NewPodClient(f).Create(ctx, pod)

	ginkgo.By("Waiting for all containers to be running")
	err := e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod)
	framework.ExpectNoError(err, "Failed to start pod")

	ginkgo.By("Verifying all containers stay running with no restarts")
	gomega.Consistently(ctx, func(ctx context.Context) error {
		updatedPod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		// The pod uses RestartPolicy Never, so a failed liveness probe kills the
		// container without incrementing RestartCount. Also require every container
		// status to be present and Running to catch probe-induced kills.
		if len(updatedPod.Status.ContainerStatuses) != len(pod.Spec.Containers) {
			return fmt.Errorf("expected %d container statuses, got %d", len(pod.Spec.Containers), len(updatedPod.Status.ContainerStatuses))
		}
		for _, containerStatus := range updatedPod.Status.ContainerStatuses {
			if containerStatus.RestartCount > 0 {
				return fmt.Errorf("container %s restarted %d times", containerStatus.Name, containerStatus.RestartCount)
			}
			switch {
			case containerStatus.State.Waiting != nil:
				return fmt.Errorf("container %s is waiting: %s", containerStatus.Name, containerStatus.State.Waiting.Reason)
			case containerStatus.State.Terminated != nil:
				return fmt.Errorf("container %s terminated: reason=%s, exitCode=%d", containerStatus.Name, containerStatus.State.Terminated.Reason, containerStatus.State.Terminated.ExitCode)
			case containerStatus.State.Running == nil:
				return fmt.Errorf("container %s is not running", containerStatus.Name)
			}
		}
		return nil
	}, probeStressWaitTime, 1*time.Second).Should(gomega.Succeed())

	ginkgo.By("Test passed: all containers stayed running with no restarts")
}

func createPodWithHTTPProbes(numContainers int) *v1.Pod {
	return createProbeStressPod(numContainers, func(i int) (v1.Probe, []v1.ContainerPort, []string) {
		port := int32(8080 + i)
		probe := v1.Probe{
			ProbeHandler: v1.ProbeHandler{
				HTTPGet: &v1.HTTPGetAction{
					Path: "/",
					Port: intstr.FromInt(int(port)),
				},
			},
			PeriodSeconds:    probeStressPeriodSeconds,
			TimeoutSeconds:   1,
			SuccessThreshold: 1,
			FailureThreshold: 3,
		}
		ports := []v1.ContainerPort{{ContainerPort: port, Protocol: v1.ProtocolTCP}}
		args := []string{"netexec", fmt.Sprintf("--http-port=%d", port)}
		return probe, ports, args
	})
}

func createPodWithTCPProbes(numContainers int) *v1.Pod {
	return createProbeStressPod(numContainers, func(i int) (v1.Probe, []v1.ContainerPort, []string) {
		port := int32(8080 + i)
		probe := v1.Probe{
			ProbeHandler: v1.ProbeHandler{
				TCPSocket: &v1.TCPSocketAction{
					Port: intstr.FromInt(int(port)),
				},
			},
			PeriodSeconds:    probeStressPeriodSeconds,
			TimeoutSeconds:   1,
			SuccessThreshold: 1,
			FailureThreshold: 3,
		}
		ports := []v1.ContainerPort{{ContainerPort: port, Protocol: v1.ProtocolTCP}}
		args := []string{"netexec", fmt.Sprintf("--http-port=%d", port)}
		return probe, ports, args
	})
}

func createPodWithGRPCProbes(numContainers int) *v1.Pod {
	return createProbeStressPod(numContainers, func(i int) (v1.Probe, []v1.ContainerPort, []string) {
		port := int32(5000 + i)
		probe := v1.Probe{
			ProbeHandler: v1.ProbeHandler{
				GRPC: &v1.GRPCAction{
					Port: port,
				},
			},
			PeriodSeconds:    probeStressPeriodSeconds,
			TimeoutSeconds:   1,
			SuccessThreshold: 1,
			FailureThreshold: 3,
		}
		ports := []v1.ContainerPort{{ContainerPort: port, Protocol: v1.ProtocolTCP}}
		args := []string{"grpc-health-checking", fmt.Sprintf("--port=%d", port)}
		return probe, ports, args
	})
}

func createProbeStressPod(numContainers int, generator func(i int) (v1.Probe, []v1.ContainerPort, []string)) *v1.Pod {
	podName := "probe-stress-" + string(uuid.NewUUID())
	containers := make([]v1.Container, numContainers)

	for i := range numContainers {
		probe, ports, args := generator(i)
		containers[i] = v1.Container{
			Name:            fmt.Sprintf("container-%d", i),
			Image:           imageutils.GetE2EImage(imageutils.Agnhost),
			Args:            args,
			Ports:           ports,
			LivenessProbe:   &probe,
			ImagePullPolicy: v1.PullIfNotPresent,
		}
	}

	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			Containers:    containers,
			RestartPolicy: v1.RestartPolicyNever,
		},
	}
}
