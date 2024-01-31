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
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

const (
	probeTestInitialDelaySeconds = 15
	numTestPods                  = 1
	numContainers                = 400
	containerPort                = 2000
)

type containerConfig struct {
	Name         string
	ProbeType    string
	ContainerIdx int
}

var _ = SIGDescribe("Stress test probes", framework.WithSerial(), framework.WithSlow(), framework.WithDisruptive(), func() {
	f := framework.NewDefaultFramework("stress-test-probe")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelBaseline

	/*
		Release: v1.30
		Testname: Pod liveness probe stress test, using http endpoint, no restart
		Description: A Pod is created with MANY containers with liveness probe on http endpoint /. Liveness probe MUST not fail to check health and the restart count should remain 0 for all containers.
	*/
	ginkgo.It("should *not be* restarted with http liveness probe stress test with MANY containers", func(ctx context.Context) {
		livenessProbe := v1.Probe{
			ProbeHandler:        httpGetHandler("/", containerPort),
			InitialDelaySeconds: 15,
			FailureThreshold:    1,
		}
		pod := livenessPodSpec("http", f.Namespace.Name, livenessProbe)
		runLivenessTest(ctx, f, pod)
	})

	/*
		Release: v1.30
		Testname: Pod liveness probe stress test, using tcp socket, no restart
		Description: A Pod is created with MANY containers with liveness probe on tcp socket 8080. Liveness probe MUST not fail to check health and the restart count should remain 0 for all containers.
	*/
	ginkgo.It("should *not* be restarted with a tcp:8080 liveness probe stress test with MANY containers", func(ctx context.Context) {
		livenessProbe := v1.Probe{
			ProbeHandler:        tcpSocketHandler(containerPort),
			InitialDelaySeconds: 15,
			FailureThreshold:    1,
		}
		pod := livenessPodSpec("tcp", f.Namespace.Name, livenessProbe)
		runLivenessTest(ctx, f, pod)
	})

	/*
		Release: v1.30
		Testname: Pod liveness probe stress test, using grpc call, no restart
		Description: A Pod is created with MANY containers with liveness probe on grpc service. Liveness probe MUST not fail to check health and the restart count should remain 0 for all containers.
	*/
	ginkgo.It("should *not* be restarted with a GRPC liveness probe stress test with MANY containers", func(ctx context.Context) {
		livenessProbe := v1.Probe{
			ProbeHandler: v1.ProbeHandler{
				GRPC: &v1.GRPCAction{
					Port:    containerPort,
					Service: nil,
				},
			},
			InitialDelaySeconds: probeTestInitialDelaySeconds,
			TimeoutSeconds:      5, // default 1s can be pretty aggressive in CI environments with low resources
			FailureThreshold:    1,
		}
		pod := livenessPodSpec("grpc", f.Namespace.Name, livenessProbe)
		runLivenessTest(ctx, f, pod)
	})
})

func livenessPodSpec(probeType, namespace string, livenessProbe v1.Probe) *v1.Pod {
	pod := e2epod.NewAgnhostPod(namespace, "liveness-stress-"+string(uuid.NewUUID()), nil, nil, nil)
	pod.ObjectMeta.Labels = map[string]string{"test": "liveness-stress"}
	pod.Spec.Containers[0].LivenessProbe = &livenessProbe
	pod.Spec.Containers = append(pod.Spec.Containers, make([]v1.Container, numContainers-1)...)
	for idx := 0; idx < numContainers; idx++ {
		cc := containerConfig{
			Name:         "agnhost-container-" + strconv.Itoa(idx+1),
			ProbeType:    probeType,
			ContainerIdx: idx,
		}
		instance := *pod.Spec.Containers[0].DeepCopy()
		pod.Spec.Containers[idx] = newContainer(cc, instance)
	}
	return pod
}

func newContainer(config containerConfig, instance v1.Container) v1.Container {
	newContainerPort := config.ContainerIdx + containerPort + 1
	instance.Name = config.Name
	instance.Ports = []v1.ContainerPort{{ContainerPort: int32(newContainerPort)}}

	switch config.ProbeType {
	case "tcp":
		instance.LivenessProbe.TCPSocket.Port = intstr.FromInt(newContainerPort)
		instance.Args = []string{"serve-hostname", "--tcp", "--port", strconv.Itoa(newContainerPort), "--http=false"}
	case "http":
		instance.LivenessProbe.HTTPGet.Port = intstr.FromInt(newContainerPort)
		instance.Args = []string{"serve-hostname", "--http", "--port", strconv.Itoa(newContainerPort)}
	case "grpc":
		instance.LivenessProbe.GRPC.Port = int32(newContainerPort)
		instance.Args = []string{"grpc-health-checking", "--port", strconv.Itoa(newContainerPort), "--http-port", strconv.Itoa(newContainerPort + containerPort)}
	}

	return instance
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

// runLivenessTest verifies the number of restarts.
func runLivenessTest(ctx context.Context, f *framework.Framework, pod *v1.Pod) {
	podClient := e2epod.NewPodClient(f)
	ns := f.Namespace.Name
	numContainersInPod := len(pod.Spec.Containers)
	gomega.Expect(numContainersInPod).To(gomega.Equal(numContainers), "pod should have a %v countainers but have %v", numContainers, numContainersInPod)
	// At the end of the test, clean up by removing the pod.
	ginkgo.DeferCleanup(func(ctx context.Context) error {
		ginkgo.By("deleting the pod")
		return podClient.Delete(ctx, pod.Name, *metav1.NewDeleteOptions(0))
	})
	ginkgo.By(fmt.Sprintf("Creating pod %s in namespace %s", pod.Name, ns))
	podClient.Create(ctx, pod)

	// Wait until the pod is running.
	framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod),
		fmt.Sprintf("starting pod %s in namespace %s", pod.Name, ns))
	framework.Logf("Started pod %s in namespace %s", pod.Name, ns)

	retries := int(framework.DefaultObservationTimeout.Seconds() / 10)
	for attempt := 0; attempt < retries; attempt++ {
		// Check the pod's current state and verify that restartCount is present.
		ginkgo.By("checking the pod's current state and verifying that restartCount is present")
		pod, err := podClient.Get(ctx, pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, fmt.Sprintf("getting pod %s in namespace %s", pod.Name, ns))
		restartCount := e2epod.GetRestartCount(pod)

		if restartCount != 0 {
			framework.Failf("Restart count of pod %s/%s is now %d (attempt %d)",
				ns, pod.Name, restartCount, attempt+1)
		}

		time.Sleep(10 * time.Second)
	}
}
