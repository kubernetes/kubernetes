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

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
)

type testCase struct {
	podSpec          *v1.Pod
	oomContainerName string
}

var _ = SIGDescribe("ndixita OOMKiller BestEffort Pod [LinuxOnly] [Serial]", func() {
	f := framework.NewDefaultFramework("oomkiller-test")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

	testCase := getOOMTestCase("besteffort")
	ginkgo.Context("", func() {
		var oldCfg *config.KubeletConfiguration
		var err error
		ginkgo.BeforeEach(func() {
			oldCfg, err = getCurrentKubeletConfig(context.TODO())
			framework.ExpectNoError(err)
			newCfg := oldCfg.DeepCopy()
			if newCfg.KubeReserved == nil {
				newCfg.KubeReserved = map[string]string{}
			}
			newCfg.KubeReserved["memory"] = "1500Mi"
			updateKubeletConfig(context.TODO(), f, newCfg, true)

			ginkgo.By("setting up the pod to be used in the test")
			e2epod.NewPodClient(f).Create(context.TODO(), testCase.podSpec)
		})

		ginkgo.It("The containers terminated by OOM killer should have the reason set to OOMKilled", func() {
			verifyReasonForOOMKilledContainer(f, testCase)
		})

		ginkgo.AfterEach(func() {
			updateKubeletConfig(context.TODO(), f, oldCfg, true)
			ginkgo.By(fmt.Sprintf("deleting pod: %s", testCase.podSpec.Name))
			e2epod.NewPodClient(f).DeleteSync(context.TODO(), testCase.podSpec.Name, metav1.DeleteOptions{}, framework.PodDeleteTimeout)
		})
	})
})

var _ = SIGDescribe("OOMKiller Guaranteed Pod [LinuxOnly] [NodeConformance]", func() {
	f := framework.NewDefaultFramework("oomkiller-test")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

	testCase := getOOMTestCase("guaranteed")

	ginkgo.Context("", func() {
		ginkgo.BeforeEach(func() {
			ginkgo.By("setting up the pod to be used in the test")
			e2epod.NewPodClient(f).Create(context.TODO(), testCase.podSpec)
		})
		ginkgo.It("The containers terminated by OOM killer should have the reason set to OOMKilled", func() {
			verifyReasonForOOMKilledContainer(f, testCase)
		})

		ginkgo.AfterEach(func() {
			ginkgo.By(fmt.Sprintf("deleting pod: %s", testCase.podSpec.Name))
			e2epod.NewPodClient(f).DeleteSync(context.TODO(), testCase.podSpec.Name, metav1.DeleteOptions{}, framework.PodDeleteTimeout)
		})
	})
})

func verifyReasonForOOMKilledContainer(f *framework.Framework, testCase *testCase) {
	ginkgo.By("Waiting for the pod to be failed")
	e2epod.WaitForPodTerminatedInNamespace(context.TODO(), f.ClientSet, testCase.podSpec.Name, "", f.Namespace.Name)

	ginkgo.By("Fetching the latest pod status")
	pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(context.TODO(), testCase.podSpec.Name, metav1.GetOptions{})
	framework.ExpectNoError(err, "Failed to get the recent pod object for name: %q", pod.Name)

	ginkgo.By("Verifying the OOM target container has the expected reason")
	container := e2epod.FindContainerStatusInPod(pod, testCase.oomContainerName)
	if container == nil {
		framework.Failf("OOM target pod %q, container %q does not have the expected state terminated", pod.Name, testCase.oomContainerName)
	}
	if container.State.Terminated == nil {
		framework.Failf("OOM target pod %q, container %q is not in the terminated state", pod.Name, container.Name)
	}
	framework.ExpectEqual(container.State.Terminated.ExitCode, int32(137),
		fmt.Sprintf("pod: %q, container: %q has unexpected exitCode: %q", pod.Name, container.Name, container.State.Terminated.ExitCode))
	framework.ExpectEqual(container.State.Terminated.Reason, "OOMKilled",
		fmt.Sprintf("pod: %q, container: %q has unexpected reason: %q", pod.Name, container.Name, container.State.Terminated.Reason))
}

func getOOMTestCase(qos string) *testCase {
	var container *v1.Container
	containerName := fmt.Sprintf("oomkill-%s-container", qos)
	switch qos {
	case "besteffort":
		container = getBestEffortContainer(containerName)
	default:
		container = getGuaranteedContainer(containerName)
	}

	return &testCase{
		podSpec: &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: fmt.Sprintf("oomkill-%s-pod", qos),
			},
			Spec: v1.PodSpec{
				RestartPolicy: v1.RestartPolicyNever,
				Containers: []v1.Container{
					*container,
				},
			},
		},
		oomContainerName: containerName,
	}
}

func getGuaranteedContainer(name string) *v1.Container {
	return &v1.Container{
		Name:  name,
		Image: busyboxImage,
		Command: []string{
			"sh",
			"-c",
			// use the dd tool to attempt to allocate 20M in a block which exceeds the limit.
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

func getBestEffortContainer(name string) *v1.Container {
	return &v1.Container{
		Name:  name,
		Image: busyboxImage,
		Command: []string{
			"sh",
			"-c",
			// use the dd tool to attempt to allocate 10G in a block which exceeds the node capacity.
			"sleep 5 && dd if=/dev/zero of=/dev/null iflag=fullblock count=10 bs=10G",
		},
	}
}
