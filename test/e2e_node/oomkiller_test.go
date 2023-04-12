/*
Copyright 2022 The Kubernetes Authors.

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

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
)

type testCase struct {
	podSpec                *v1.Pod
	oomTargetContainerName string
}

const PodOOMKilledTimeout = 2 * time.Minute

var _ = SIGDescribe("OOMKiller [LinuxOnly] [NodeConformance]", func() {
	f := framework.NewDefaultFramework("oomkiller-test")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

	containerName := "oomkill-target-container"
	oomPodSpec := getOOMTargetPod("oomkill-target-pod", containerName)
	runOomKillerTest(f, testCase{podSpec: oomPodSpec, oomTargetContainerName: containerName})
})

func runOomKillerTest(f *framework.Framework, testCase testCase) {
	ginkgo.Context("", func() {
		ginkgo.BeforeEach(func() {
			ginkgo.By("setting up the pod to be used in the test")
			e2epod.NewPodClient(f).Create(context.TODO(), testCase.podSpec)
		})

		ginkgo.It("The containers terminated by OOM killer should have the reason set to OOMKilled", func() {

			ginkgo.By("Waiting for the pod to be failed")
			e2epod.WaitForPodTerminatedInNamespace(context.TODO(), f.ClientSet, testCase.podSpec.Name, "", f.Namespace.Name)

			ginkgo.By("Fetching the latest pod status")
			pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(context.TODO(), testCase.podSpec.Name, metav1.GetOptions{})
			framework.ExpectNoError(err, "Failed to get the recent pod object for name: %q", pod.Name)

			ginkgo.By("Verifying the OOM target container has the expected reason")
			verifyReasonForOOMKilledContainer(pod, testCase.oomTargetContainerName)
		})

		ginkgo.AfterEach(func() {
			ginkgo.By(fmt.Sprintf("deleting pod: %s", testCase.podSpec.Name))
			e2epod.NewPodClient(f).DeleteSync(context.TODO(), testCase.podSpec.Name, metav1.DeleteOptions{}, framework.PodDeleteTimeout)
		})
	})
}

func verifyReasonForOOMKilledContainer(pod *v1.Pod, oomTargetContainerName string) {
	container := e2epod.FindContainerStatusInPod(pod, oomTargetContainerName)
	if container == nil {
		framework.Failf("OOM target pod %q, container %q does not have the expected state terminated", pod.Name, container.Name)
	}
	if container.State.Terminated == nil {
		framework.Failf("OOM target pod %q, container %q is not in the terminated state", pod.Name, container.Name)
	}
	framework.ExpectEqual(container.State.Terminated.ExitCode, int32(137),
		fmt.Sprintf("pod: %q, container: %q has unexpected exitCode: %q", pod.Name, container.Name, container.State.Terminated.ExitCode))
	framework.ExpectEqual(container.State.Terminated.Reason, "OOMKilled",
		fmt.Sprintf("pod: %q, container: %q has unexpected reason: %q", pod.Name, container.Name, container.State.Terminated.Reason))
}

func getOOMTargetPod(podName string, ctnName string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers: []v1.Container{
				getOOMTargetContainer(ctnName),
			},
		},
	}
}

func getOOMTargetContainer(name string) v1.Container {
	return v1.Container{
		Name:  name,
		Image: busyboxImage,
		Command: []string{
			"sh",
			"-c",
			// use the dd tool to attempt to allocate 20M in a block which exceeds the limit
			"sleep 5 && dd if=/dev/zero of=/dev/null bs=20M",
		},
		Resources: v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceMemory: resource.MustParse("15Mi"),
			},
			Limits: v1.ResourceList{
				v1.ResourceMemory: resource.MustParse("15Mi"),
			},
		},
	}
}
