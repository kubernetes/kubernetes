/*
Copyright 2023 The Kubernetes Authors.
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
	"strconv"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/uuid"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

const (
	probeTestInitialDelaySeconds = 15
	numTestPods                  = 1
	numContainers                = 100
	defaultObservationTimeout    = time.Minute * 4
)

var _ = SIGDescribe("Stress test probes", func() {
	f := framework.NewDefaultFramework("stress-test-probe")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelBaseline

	/*
		Release: v1.23
		Testname: Pod liveness probe stress test, using http endpoint, no restart
		Description: A Pod is created with MANY containers with liveness probe on http endpoint /. Liveness probe MUST not fail to check health and the restart count should remain 0 for all containers.
	*/
	ginkgo.It("should *not be* restarted with http liveness probe stress test with MANY containers", func(ctx context.Context) {
		livenessProbe := v1.Probe{
			ProbeHandler:        httpGetHandler("/", 2000),
			InitialDelaySeconds: 15,
			FailureThreshold:    1,
		}
		ports := []v1.ContainerPort{{ContainerPort: int32(2000)}}
		probeArgs := []string{"serve-hostname", "--http", "--port", ""}
		pod := livenessPodSpec("http", f.Namespace.Name, livenessProbe, ports, probeArgs...)
		RunLivenessTest(ctx, f, pod, defaultObservationTimeout)
	})

	/*
		Release: v1.23
		Testname: Pod liveness probe stress test, using tcp socket, no restart
		Description: A Pod is created with MANY containers with liveness probe on tcp socket 8080. Liveness probe MUST not fail to check health and the restart count should remain 0 for all containers.
	*/
	ginkgo.It("should *not* be restarted with a tcp:8080 liveness probe stress test with MANY containers", func(ctx context.Context) {
		livenessProbe := v1.Probe{
			ProbeHandler:        tcpSocketHandler(2000),
			InitialDelaySeconds: 15,
			FailureThreshold:    1,
		}
		ports := []v1.ContainerPort{{ContainerPort: int32(2000)}}
		probeArgs := []string{"serve-hostname", "--tcp", "--port", "", "--http=false"}
		pod := livenessPodSpec("tcp", f.Namespace.Name, livenessProbe, ports, probeArgs...)
		RunLivenessTest(ctx, f, pod, defaultObservationTimeout)
	})

	/*
		Release: v1.23
		Testname: Pod liveness probe stress test, using grpc call, no restart
		Description: A Pod is created with MANY containers with liveness probe on grpc service. Liveness probe MUST not fail to check health and the restart count should remain 0 for all containers.
	*/
	ginkgo.It("should *not* be restarted with a GRPC liveness probe stress test with MANY containers", func(ctx context.Context) {
		livenessProbe := v1.Probe{
			ProbeHandler: v1.ProbeHandler{
				GRPC: &v1.GRPCAction{
					Port:    2000,
					Service: nil,
				},
			},
			InitialDelaySeconds: probeTestInitialDelaySeconds,
			TimeoutSeconds:      5, // default 1s can be pretty aggressive in CI environments with low resources
			FailureThreshold:    1,
		}
		ports := []v1.ContainerPort{{ContainerPort: int32(2000)}}
		probeArgs := []string{"grpc-health-checking", "--port", "", "--http-port", ""}
		pod := livenessPodSpec("grpc", f.Namespace.Name, livenessProbe, ports, probeArgs...)
		RunLivenessTest(ctx, f, pod, defaultObservationTimeout)
	})
})

func livenessPodSpec(probeType, namespace string, livenessProbe v1.Probe, ports []v1.ContainerPort, args ...string) *v1.Pod {
	pod := e2epod.NewAgnhostPod(namespace, "liveness-stress-"+string(uuid.NewUUID()), nil, nil, ports, args...)
	pod.ObjectMeta.Labels = map[string]string{"test": "liveness-stress"}
	pod.Spec.Containers[0].LivenessProbe = &livenessProbe
	pod.Spec.Containers[0].Name = "agnhost-container-0"
	pod.Spec.Containers = append(pod.Spec.Containers, make([]v1.Container, numContainers-1)...)
	for idx, _ := range make([]int, numContainers) {
		containerPort := idx + 2001
		pod.Spec.Containers[idx] = *pod.Spec.Containers[0].DeepCopy()
		pod.Spec.Containers[idx].Name = "agnhost-container-" + strconv.Itoa(idx+1)
		pod.Spec.Containers[idx].Ports = []v1.ContainerPort{{ContainerPort: int32(containerPort)}}
		switch probeType {
		case "tcp":
			pod.Spec.Containers[idx].LivenessProbe.TCPSocket.Port = intstr.FromInt(containerPort)
			pod.Spec.Containers[idx].Args[3] = strconv.Itoa(containerPort)
		case "http":
			pod.Spec.Containers[idx].LivenessProbe.HTTPGet.Port = intstr.FromInt(containerPort)
			pod.Spec.Containers[idx].Args[3] = strconv.Itoa(containerPort)
		case "grpc":
			pod.Spec.Containers[idx].LivenessProbe.GRPC.Port = int32(containerPort)
			pod.Spec.Containers[idx].Args[2] = strconv.Itoa(containerPort)
			pod.Spec.Containers[idx].Args[4] = strconv.Itoa(containerPort + 2000)
		}
	}
	return pod
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

// RunLivenessTest verifies the number of restarts.
func RunLivenessTest(ctx context.Context, f *framework.Framework, pod *v1.Pod, timeout time.Duration) {
	deadline := time.Now().Add(timeout)
	podClient := e2epod.NewPodClient(f)
	ns := f.Namespace.Name
	gomega.Expect(pod.Spec.Containers).NotTo(gomega.BeEmpty())
	// At the end of the test, clean up by removing the pod.
	ginkgo.DeferCleanup(func(ctx context.Context) error {
		ginkgo.By("deleting the pod")
		return podClient.Delete(ctx, pod.Name, *metav1.NewDeleteOptions(0))
	})
	ginkgo.By(fmt.Sprintf("Creating pod %s in namespace %s", pod.Name, ns))
	podClient.Create(ctx, pod)

	// Wait until the pod is not pending. (Here we need to check for something other than
	// 'Pending' other than checking for 'Running', since when failures occur, we go to
	// 'Terminated' which can cause indefinite blocking.)
	framework.ExpectNoError(e2epod.WaitForPodNotPending(ctx, f.ClientSet, ns, pod.Name),
		fmt.Sprintf("starting pod %s in namespace %s", pod.Name, ns))
	framework.Logf("Started pod %s in namespace %s", pod.Name, ns)

	// Check the pod's current state and verify that restartCount is present.
	ginkgo.By("checking the pod's current state and verifying that restartCount is present")
	pod, err := podClient.Get(ctx, pod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err, fmt.Sprintf("getting pod %s in namespace %s", pod.Name, ns))

cron:
	for start := time.Now(); time.Now().Before(deadline); time.Sleep(60 * time.Second) {
		pod, err = podClient.Get(ctx, pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, fmt.Sprintf("getting pod %s", pod.Name))
		for idx, _ := range [numContainers]int{} {
			containerName := "agnhost-container-" + strconv.Itoa(idx)

			restartCount := podutil.GetExistingContainerStatus(pod.Status.ContainerStatuses, containerName).RestartCount

			if restartCount != int32(0) {
				framework.Failf("Restart count of pod's container %s/%s/%s is now %d (%v elapsed)",
					ns, pod.Name, containerName, restartCount, time.Since(start))
				break cron
			}
		}
	}
}
