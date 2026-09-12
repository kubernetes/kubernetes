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

package node

import (
	"context"
	"fmt"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

// numStressProbeContainers is the number of containers per pod in probe stress tests.
// High enough to expose port-exhaustion regressions (issue #89898) while
// remaining schedulable on a single node in CI.
const numStressProbeContainers = 20

// probeStressDuration is how long we let all probes fire before declaring success.
// 1 probe/sec × 20 containers × 120 s = 2400 probe connections; enough to stress
// the ephemeral port / conntrack path.
const probeStressDuration = 2 * time.Minute

// probeStressPollInterval is how often we re-check restart counts while waiting.
const probeStressPollInterval = 10 * time.Second

var _ = SIGDescribe("Probe stress", framework.WithSerial(), framework.WithSlow(), func() {
	f := framework.NewDefaultFramework("probe-stress")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	// Each It block creates one pod with numStressProbeContainers containers, each
	// running a 1-second liveness probe on a unique port, then asserts zero restarts
	// for the entire probeStressDuration.  Regression test for issue #89898 (TCP
	// ephemeral-port / conntrack exhaustion from rapid kubelet probe connections).

	ginkgo.It("should not restart any container with concurrent HTTP liveness probes", func(ctx context.Context) {
		pod := httpStressPod(f.Namespace.Name)
		runProbeStressTest(ctx, f, pod)
	})

	ginkgo.It("should not restart any container with concurrent TCP liveness probes", func(ctx context.Context) {
		pod := tcpStressPod(f.Namespace.Name)
		runProbeStressTest(ctx, f, pod)
	})

	ginkgo.It("should not restart any container with concurrent gRPC liveness probes", func(ctx context.Context) {
		pod := grpcStressPod(f.Namespace.Name)
		runProbeStressTest(ctx, f, pod)
	})
})

// runProbeStressTest creates the pod, waits for it to be running, then uses
// Consistently to assert that no container accumulates restarts over
// probeStressDuration.
func runProbeStressTest(ctx context.Context, f *framework.Framework, pod *v1.Pod) {
	podClient := e2epod.NewPodClient(f)

	ginkgo.By(fmt.Sprintf("creating pod %s with %d containers", pod.Name, len(pod.Spec.Containers)))
	pod = podClient.Create(ctx, pod)
	ginkgo.DeferCleanup(func(ctx context.Context) {
		_ = podClient.Delete(ctx, pod.Name, *metav1.NewDeleteOptions(0))
	})

	ginkgo.By("waiting for all containers to be running")
	framework.ExpectNoError(
		e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod),
		"pod %s/%s did not reach Running", f.Namespace.Name, pod.Name,
	)

	ginkgo.By(fmt.Sprintf("observing probes for %s; any restart is a failure", probeStressDuration))
	gomega.Consistently(ctx, func(ctx context.Context) error {
		p, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		for _, cs := range p.Status.ContainerStatuses {
			if cs.RestartCount > 0 {
				return fmt.Errorf("container %q restarted %d time(s): liveness probe is firing false positives",
					cs.Name, cs.RestartCount)
			}
		}
		return nil
	}, probeStressDuration, probeStressPollInterval).Should(gomega.Succeed())
}

// httpStressPod creates a pod with numStressProbeContainers containers.  Each
// container runs agnhost netexec on a distinct port so they do not conflict
// within the shared pod network namespace.  The kubelet probes each via
// PodIP:containerPort, stressing the ephemeral-port / conntrack path.
func httpStressPod(namespace string) *v1.Pod {
	return stressPod(namespace, "probe-stress-http", func(i int) (v1.Probe, []v1.ContainerPort, []string) {
		port := int32(8080 + i)
		probe := v1.Probe{
			ProbeHandler: v1.ProbeHandler{
				HTTPGet: &v1.HTTPGetAction{
					Path: "/",
					Port: intstr.FromInt32(port),
				},
			},
			PeriodSeconds:    1,
			TimeoutSeconds:   1,
			SuccessThreshold: 1,
			FailureThreshold: 3,
		}
		ports := []v1.ContainerPort{{ContainerPort: port}}
		args := []string{"netexec", fmt.Sprintf("--http-port=%d", port)}
		return probe, ports, args
	})
}

// tcpStressPod creates a pod with numStressProbeContainers containers.  Each
// container runs agnhost netexec on a distinct port with a TCP liveness probe.
// TCP probes go through the same connect-close cycle that exhausted conntrack
// entries in issue #89898.
func tcpStressPod(namespace string) *v1.Pod {
	return stressPod(namespace, "probe-stress-tcp", func(i int) (v1.Probe, []v1.ContainerPort, []string) {
		port := int32(8080 + i)
		probe := v1.Probe{
			ProbeHandler: v1.ProbeHandler{
				TCPSocket: &v1.TCPSocketAction{
					Port: intstr.FromInt32(port),
				},
			},
			PeriodSeconds:    1,
			TimeoutSeconds:   1,
			SuccessThreshold: 1,
			FailureThreshold: 3,
		}
		ports := []v1.ContainerPort{{ContainerPort: port}}
		args := []string{"netexec", fmt.Sprintf("--http-port=%d", port)}
		return probe, ports, args
	})
}

// grpcStressPod creates a pod with numStressProbeContainers containers.  Each
// container runs agnhost grpc-health-checking on a distinct port with a gRPC
// liveness probe.
func grpcStressPod(namespace string) *v1.Pod {
	return stressPod(namespace, "probe-stress-grpc", func(i int) (v1.Probe, []v1.ContainerPort, []string) {
		port := int32(5000 + i)
		probe := v1.Probe{
			ProbeHandler: v1.ProbeHandler{
				GRPC: &v1.GRPCAction{
					Port: port,
				},
			},
			PeriodSeconds:    1,
			TimeoutSeconds:   1,
			SuccessThreshold: 1,
			FailureThreshold: 3,
		}
		ports := []v1.ContainerPort{{ContainerPort: port}}
		args := []string{"grpc-health-checking", fmt.Sprintf("--port=%d", port)}
		return probe, ports, args
	})
}

// stressPod builds the pod skeleton.  generator(i) returns the liveness probe,
// container ports, and agnhost args for container i.  Each container gets a
// unique name so the kubelet tracks its restart count independently.
func stressPod(namespace, namePrefix string, generator func(i int) (v1.Probe, []v1.ContainerPort, []string)) *v1.Pod {
	containers := make([]v1.Container, numStressProbeContainers)
	for i := range containers {
		probe, ports, args := generator(i)
		containers[i] = v1.Container{
			Name:          fmt.Sprintf("container-%d", i),
			Image:         imageutils.GetE2EImage(imageutils.Agnhost),
			Args:          args,
			Ports:         ports,
			LivenessProbe: &probe,
		}
	}
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      namePrefix + "-" + string(uuid.NewUUID()),
			Namespace: namespace,
		},
		Spec: v1.PodSpec{
			Containers:    containers,
			RestartPolicy: v1.RestartPolicyAlways,
		},
	}
}
