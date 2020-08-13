/*
Copyright 2016 The Kubernetes Authors.

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
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

var _ = framework.KubeDescribe("Docker Containers", func() {
	f := framework.NewDefaultFramework("containers")

	/*
		Release: v1.9
		Testname: Docker containers, without command and arguments
		Description: Default command and arguments from the docker image entrypoint MUST be used when Pod does not specify the container command
	*/
	framework.ConformanceIt("should use the image defaults if command and args are blank [NodeConformance]", func() {
		pod := f.PodClient().Create(entrypointTestPod())
		err := e2epod.WaitForPodNameRunningInNamespace(f.ClientSet, pod.Name, f.Namespace.Name)
		framework.ExpectNoError(err, "Expected pod %q to be running, got error: %v", pod.Name, err)

		pollLogs := func() (string, error) {
			return e2epod.GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, containerName)
		}

		// The agnhost's image default entrypoint / args are: "/agnhost pause"
		// which will print out "Paused".
		gomega.Eventually(pollLogs, 3, framework.Poll).Should(gomega.ContainSubstring("Paused"))
	})

	/*
		Release: v1.9
		Testname: Docker containers, with arguments
		Description: Default command and  from the docker image entrypoint MUST be used when Pod does not specify the container command but the arguments from Pod spec MUST override when specified.
	*/
	framework.ConformanceIt("should be able to override the image's default arguments (docker cmd) [NodeConformance]", func() {
		pod := entrypointTestPod()
		pod.Spec.Containers[0].Args = []string{"entrypoint-tester", "override", "arguments"}

		f.TestContainerOutput("override arguments", pod, 0, []string{
			"[/agnhost entrypoint-tester override arguments]",
		})
	})

	// Note: when you override the entrypoint, the image's arguments (docker cmd)
	// are ignored.
	/*
		Release: v1.9
		Testname: Docker containers, with command
		Description: Default command from the docker image entrypoint MUST NOT be used when Pod specifies the container command.  Command from Pod spec MUST override the command in the image.
	*/
	framework.ConformanceIt("should be able to override the image's default command (docker entrypoint) [NodeConformance]", func() {
		pod := entrypointTestPod()
		pod.Spec.Containers[0].Command = []string{"/agnhost-2", "entrypoint-tester"}

		f.TestContainerOutput("override command", pod, 0, []string{
			"[/agnhost-2 entrypoint-tester]",
		})
	})

	/*
		Release: v1.9
		Testname: Docker containers, with command and arguments
		Description: Default command and arguments from the docker image entrypoint MUST NOT be used when Pod specifies the container command and arguments.  Command and arguments from Pod spec MUST override the command and arguments in the image.
	*/
	framework.ConformanceIt("should be able to override the image's default command and arguments [NodeConformance]", func() {
		pod := entrypointTestPod()
		pod.Spec.Containers[0].Command = []string{"/agnhost-2"}
		pod.Spec.Containers[0].Args = []string{"entrypoint-tester", "override", "arguments"}

		f.TestContainerOutput("override all", pod, 0, []string{
			"[/agnhost-2 entrypoint-tester override arguments]",
		})
	})
})

const testContainerName = "test-container"

// Return a prototypical entrypoint test pod
func entrypointTestPod() *v1.Pod {
	podName := "client-containers-" + string(uuid.NewUUID())

	one := int64(1)
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  testContainerName,
					Image: imageutils.GetE2EImage(imageutils.Agnhost),
				},
			},
			RestartPolicy:                 v1.RestartPolicyNever,
			TerminationGracePeriodSeconds: &one,
		},
	}
}
