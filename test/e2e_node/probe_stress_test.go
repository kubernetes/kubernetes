//go:build linux
// +build linux

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
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.Context("HTTP liveness probes", func() {
		ginkgo.It("should not cause unexpected container restarts under load [Serial]", func(ctx context.Context) {
			pod := createPodWithHTTPProbes(probeStressNumContainers)
			runProbeStressTest(ctx, f, pod)
		})
	})

	ginkgo.Context("TCP liveness probes", func() {
		ginkgo.It("should not cause unexpected container restarts under load [Serial]", func(ctx context.Context) {
			pod := createPodWithTCPProbes(probeStressNumContainers)
			runProbeStressTest(ctx, f, pod)
		})
	})

	ginkgo.Context("gRPC liveness probes", func() {
		ginkgo.It("should not cause unexpected container restarts under load [Serial]", func(ctx context.Context) {
			pod := createPodWithGRPCProbes(probeStressNumContainers)
			runProbeStressTest(ctx, f, pod)
		})
	})
})

// runProbeStressTest creates a pod with many containers, waits for them to be running,
// and verifies that none of the containers have restarted unexpectedly.
func runProbeStressTest(ctx context.Context, f *framework.Framework, pod *v1.Pod) {
	ginkgo.By(fmt.Sprintf("Creating pod %s with %d containers", pod.Name, len(pod.Spec.Containers)))
	pod = e2epod.NewPodClient(f).Create(ctx, pod)

	ginkgo.By("Waiting for all containers to be running")
	err := e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod)
	framework.ExpectNoError(err, "Failed to start pod")

	ginkgo.By(fmt.Sprintf("Waiting %v to observe probe behavior", probeStressWaitTime))
	time.Sleep(probeStressWaitTime)

	ginkgo.By("Verifying no containers have restarted")
	updatedPod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err, "Failed to get pod")

	for _, containerStatus := range updatedPod.Status.ContainerStatuses {
		gomega.Expect(containerStatus.RestartCount).To(gomega.BeZero(),
			"Container %s should not have restarted, but has restart count %d",
			containerStatus.Name, containerStatus.RestartCount)
	}

	ginkgo.By("Test passed: no unexpected container restarts")
}

// createPodWithHTTPProbes creates a pod with multiple containers, each with an HTTP liveness probe.
func createPodWithHTTPProbes(numContainers int) *v1.Pod {
	podName := "probe-stress-http-" + string(uuid.NewUUID())
	containers := make([]v1.Container, numContainers)

	for i := 0; i < numContainers; i++ {
		containerName := fmt.Sprintf("container-%d", i)
		port := int32(8080 + i)

		containers[i] = v1.Container{
			Name:  containerName,
			Image: imageutils.GetE2EImage(imageutils.Agnhost),
			Args:  []string{"netexec", fmt.Sprintf("--http-port=%d", port)},
			Ports: []v1.ContainerPort{
				{
					ContainerPort: port,
					Protocol:      v1.ProtocolTCP,
				},
			},
			LivenessProbe: &v1.Probe{
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
			},
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

// createPodWithTCPProbes creates a pod with multiple containers, each with a TCP liveness probe.
func createPodWithTCPProbes(numContainers int) *v1.Pod {
	podName := "probe-stress-tcp-" + string(uuid.NewUUID())
	containers := make([]v1.Container, numContainers)

	for i := 0; i < numContainers; i++ {
		containerName := fmt.Sprintf("container-%d", i)
		port := int32(8080 + i)

		containers[i] = v1.Container{
			Name:  containerName,
			Image: imageutils.GetE2EImage(imageutils.Agnhost),
			Args:  []string{"netexec", fmt.Sprintf("--http-port=%d", port)},
			Ports: []v1.ContainerPort{
				{
					ContainerPort: port,
					Protocol:      v1.ProtocolTCP,
				},
			},
			LivenessProbe: &v1.Probe{
				ProbeHandler: v1.ProbeHandler{
					TCPSocket: &v1.TCPSocketAction{
						Port: intstr.FromInt(int(port)),
					},
				},
				PeriodSeconds:    probeStressPeriodSeconds,
				TimeoutSeconds:   1,
				SuccessThreshold: 1,
				FailureThreshold: 3,
			},
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

// createPodWithGRPCProbes creates a pod with multiple containers, each with a gRPC liveness probe.
func createPodWithGRPCProbes(numContainers int) *v1.Pod {
	podName := "probe-stress-grpc-" + string(uuid.NewUUID())
	containers := make([]v1.Container, numContainers)

	for i := 0; i < numContainers; i++ {
		containerName := fmt.Sprintf("container-%d", i)
		port := int32(5000 + i)

		containers[i] = v1.Container{
			Name:  containerName,
			Image: imageutils.GetE2EImage(imageutils.Agnhost),
			Args:  []string{"grpc-health-checking", fmt.Sprintf("--port=%d", port)},
			Ports: []v1.ContainerPort{
				{
					ContainerPort: port,
					Protocol:      v1.ProtocolTCP,
				},
			},
			LivenessProbe: &v1.Probe{
				ProbeHandler: v1.ProbeHandler{
					GRPC: &v1.GRPCAction{
						Port: port,
					},
				},
				PeriodSeconds:    probeStressPeriodSeconds,
				TimeoutSeconds:   1,
				SuccessThreshold: 1,
				FailureThreshold: 3,
			},
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
