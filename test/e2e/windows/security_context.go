/*
Copyright 2019 The Kubernetes Authors.

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

package windows

import (
	"context"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
)

const runAsUserNameContainerName = "run-as-username-container"

var _ = SIGDescribe("[Feature:Windows] SecurityContext", func() {
	f := framework.NewDefaultFramework("windows-run-as-username")

	ginkgo.It("should be able create pods and run containers with a given username", func() {
		ginkgo.By("Creating 2 pods: 1 with the default user, and one with a custom one.")
		podDefault := runAsUserNamePod(nil)
		f.TestContainerOutput("check default user", podDefault, 0, []string{"ContainerUser"})

		podUserName := runAsUserNamePod(toPtr("ContainerAdministrator"))
		f.TestContainerOutput("check set user", podUserName, 0, []string{"ContainerAdministrator"})
	})

	ginkgo.It("should not be able to create pods with unknown usernames", func() {
		ginkgo.By("Creating a pod with an invalid username")
		podInvalid := f.PodClient().Create(runAsUserNamePod(toPtr("FooLish")))

		framework.Logf("Waiting for pod %s to enter the error state.", podInvalid.Name)
		framework.ExpectNoError(e2epod.WaitForPodTerminatedInNamespace(f.ClientSet, podInvalid.Name, "", f.Namespace.Name))

		podInvalid, _ = f.PodClient().Get(context.TODO(), podInvalid.Name, metav1.GetOptions{})
		podTerminatedReason := testutils.TerminatedContainers(podInvalid)[runAsUserNameContainerName]
		if podTerminatedReason != "ContainerCannotRun" && podTerminatedReason != "StartError" {
			framework.Failf("The container terminated reason was supposed to be: 'ContainerCannotRun' or 'StartError', not: '%q'", podTerminatedReason)
		}
	})

	ginkgo.It("should override SecurityContext username if set", func() {
		ginkgo.By("Creating a pod with 2 containers with different username configurations.")

		pod := runAsUserNamePod(toPtr("ContainerAdministrator"))
		pod.Spec.Containers[0].SecurityContext.WindowsOptions.RunAsUserName = toPtr("ContainerUser")
		pod.Spec.Containers = append(pod.Spec.Containers, v1.Container{
			Name:    "run-as-username-new-container",
			Image:   imageutils.GetE2EImage(imageutils.NonRoot),
			Command: []string{"cmd", "/S", "/C", "echo %username%"},
		})

		f.TestContainerOutput("check overridden username", pod, 0, []string{"ContainerUser"})
		f.TestContainerOutput("check pod SecurityContext username", pod, 1, []string{"ContainerAdministrator"})
	})
	ginkgo.It("should ignore Linux Specific SecurityContext if set", func() {
		ginkgo.By("Creating a pod with SELinux options")
		// It is sufficient to show that the pod comes up here. Since we're stripping the SELinux and other linux
		// security contexts in apiserver and not updating the pod object in the apiserver, we cannot validate the
		// the pod object to not have those security contexts. However the pod coming to running state is a sufficient
		// enough condition for us to validate since prior to https://github.com/kubernetes/kubernetes/pull/93475
		// the pod would have failed to come up.
		windowsPodWithSELinux := createTestPod(f, windowsBusyBoximage, windowsOS)
		windowsPodWithSELinux.Spec.Containers[0].Args = []string{"test-webserver-with-selinux"}
		windowsPodWithSELinux.Spec.SecurityContext = &v1.PodSecurityContext{}
		containerUserName := "ContainerAdministrator"
		windowsPodWithSELinux.Spec.SecurityContext.SELinuxOptions = &v1.SELinuxOptions{Level: "s0:c24,c9"}
		windowsPodWithSELinux.Spec.Containers[0].SecurityContext = &v1.SecurityContext{
			SELinuxOptions: &v1.SELinuxOptions{Level: "s0:c24,c9"},
			WindowsOptions: &v1.WindowsSecurityContextOptions{RunAsUserName: &containerUserName}}
		windowsPodWithSELinux.Spec.Tolerations = []v1.Toleration{{Key: "os", Value: "Windows"}}
		windowsPodWithSELinux, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(),
			windowsPodWithSELinux, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		framework.Logf("Created pod %v", windowsPodWithSELinux)
		framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(f.ClientSet, windowsPodWithSELinux.Name,
			f.Namespace.Name), "failed to wait for pod %s to be running", windowsPodWithSELinux.Name)
	})
})

func runAsUserNamePod(username *string) *v1.Pod {
	podName := "run-as-username-" + string(uuid.NewUUID())
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			NodeSelector: map[string]string{"kubernetes.io/os": "windows"},
			Containers: []v1.Container{
				{
					Name:    runAsUserNameContainerName,
					Image:   imageutils.GetE2EImage(imageutils.NonRoot),
					Command: []string{"cmd", "/S", "/C", "echo %username%"},
					SecurityContext: &v1.SecurityContext{
						WindowsOptions: &v1.WindowsSecurityContextOptions{
							RunAsUserName: username,
						},
					},
				},
			},
			SecurityContext: &v1.PodSecurityContext{
				WindowsOptions: &v1.WindowsSecurityContextOptions{
					RunAsUserName: username,
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
		},
	}
}

func toPtr(s string) *string {
	return &s
}
