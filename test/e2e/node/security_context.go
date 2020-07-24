/*
Copyright 2015 The Kubernetes Authors.

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

/* This test check that SecurityContext parameters specified at the
 * pod or the container level work as intended. These tests cannot be
 * run when the 'SecurityContextDeny' admission controller is not used
 * so they are skipped by default.
 */

package node

import (
	"context"
	"fmt"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ekubectl "k8s.io/kubernetes/test/e2e/framework/kubectl"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

func scTestPod(hostIPC bool, hostPID bool) *v1.Pod {
	podName := "security-context-" + string(uuid.NewUUID())
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        podName,
			Labels:      map[string]string{"name": podName},
			Annotations: map[string]string{},
		},
		Spec: v1.PodSpec{
			HostIPC:         hostIPC,
			HostPID:         hostPID,
			SecurityContext: &v1.PodSecurityContext{},
			Containers: []v1.Container{
				{
					Name:  "test-container",
					Image: imageutils.GetE2EImage(imageutils.BusyBox),
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
		},
	}

	return pod
}

var _ = SIGDescribe("Security Context", func() {
	f := framework.NewDefaultFramework("security-context")

	ginkgo.It("should support pod.Spec.SecurityContext.SupplementalGroups [LinuxOnly]", func() {
		pod := scTestPod(false, false)
		pod.Spec.Containers[0].Command = []string{"id", "-G"}
		pod.Spec.SecurityContext.SupplementalGroups = []int64{1234, 5678}
		groups := []string{"1234", "5678"}
		f.TestContainerOutput("pod.Spec.SecurityContext.SupplementalGroups", pod, 0, groups)
	})

	ginkgo.It("should support pod.Spec.SecurityContext.RunAsUser [LinuxOnly]", func() {
		pod := scTestPod(false, false)
		userID := int64(1001)
		pod.Spec.SecurityContext.RunAsUser = &userID
		pod.Spec.Containers[0].Command = []string{"sh", "-c", "id"}

		f.TestContainerOutput("pod.Spec.SecurityContext.RunAsUser", pod, 0, []string{
			fmt.Sprintf("uid=%v", userID),
			fmt.Sprintf("gid=%v", 0),
		})
	})

	ginkgo.It("should support pod.Spec.SecurityContext.RunAsUser And pod.Spec.SecurityContext.RunAsGroup [LinuxOnly]", func() {
		pod := scTestPod(false, false)
		userID := int64(1001)
		groupID := int64(2002)
		pod.Spec.SecurityContext.RunAsUser = &userID
		pod.Spec.SecurityContext.RunAsGroup = &groupID
		pod.Spec.Containers[0].Command = []string{"sh", "-c", "id"}

		f.TestContainerOutput("pod.Spec.SecurityContext.RunAsUser", pod, 0, []string{
			fmt.Sprintf("uid=%v", userID),
			fmt.Sprintf("gid=%v", groupID),
		})
	})

	ginkgo.It("should support container.SecurityContext.RunAsUser [LinuxOnly]", func() {
		pod := scTestPod(false, false)
		userID := int64(1001)
		overrideUserID := int64(1002)
		pod.Spec.SecurityContext.RunAsUser = &userID
		pod.Spec.Containers[0].SecurityContext = new(v1.SecurityContext)
		pod.Spec.Containers[0].SecurityContext.RunAsUser = &overrideUserID
		pod.Spec.Containers[0].Command = []string{"sh", "-c", "id"}

		f.TestContainerOutput("pod.Spec.SecurityContext.RunAsUser", pod, 0, []string{
			fmt.Sprintf("uid=%v", overrideUserID),
			fmt.Sprintf("gid=%v", 0),
		})
	})

	ginkgo.It("should support container.SecurityContext.RunAsUser And container.SecurityContext.RunAsGroup [LinuxOnly]", func() {
		pod := scTestPod(false, false)
		userID := int64(1001)
		groupID := int64(2001)
		overrideUserID := int64(1002)
		overrideGroupID := int64(2002)
		pod.Spec.SecurityContext.RunAsUser = &userID
		pod.Spec.SecurityContext.RunAsGroup = &groupID
		pod.Spec.Containers[0].SecurityContext = new(v1.SecurityContext)
		pod.Spec.Containers[0].SecurityContext.RunAsUser = &overrideUserID
		pod.Spec.Containers[0].SecurityContext.RunAsGroup = &overrideGroupID
		pod.Spec.Containers[0].Command = []string{"sh", "-c", "id"}

		f.TestContainerOutput("pod.Spec.SecurityContext.RunAsUser", pod, 0, []string{
			fmt.Sprintf("uid=%v", overrideUserID),
			fmt.Sprintf("gid=%v", overrideGroupID),
		})
	})

	ginkgo.It("should support volume SELinux relabeling [Flaky] [LinuxOnly]", func() {
		testPodSELinuxLabeling(f, false, false)
	})

	ginkgo.It("should support volume SELinux relabeling when using hostIPC [Flaky] [LinuxOnly]", func() {
		testPodSELinuxLabeling(f, true, false)
	})

	ginkgo.It("should support volume SELinux relabeling when using hostPID [Flaky] [LinuxOnly]", func() {
		testPodSELinuxLabeling(f, false, true)
	})

	ginkgo.It("should support seccomp alpha unconfined annotation on the container [Feature:Seccomp] [LinuxOnly]", func() {
		// TODO: port to SecurityContext as soon as seccomp is out of alpha
		pod := scTestPod(false, false)
		pod.Annotations[v1.SeccompContainerAnnotationKeyPrefix+"test-container"] = "unconfined"
		pod.Annotations[v1.SeccompPodAnnotationKey] = v1.SeccompProfileRuntimeDefault
		pod.Spec.Containers[0].Command = []string{"grep", "ecc", "/proc/self/status"}
		f.TestContainerOutput(v1.SeccompPodAnnotationKey, pod, 0, []string{"0"}) // seccomp disabled
	})

	ginkgo.It("should support seccomp alpha unconfined annotation on the pod [Feature:Seccomp] [LinuxOnly]", func() {
		// TODO: port to SecurityContext as soon as seccomp is out of alpha
		pod := scTestPod(false, false)
		pod.Annotations[v1.SeccompPodAnnotationKey] = "unconfined"
		pod.Spec.Containers[0].Command = []string{"grep", "ecc", "/proc/self/status"}
		f.TestContainerOutput(v1.SeccompPodAnnotationKey, pod, 0, []string{"0"}) // seccomp disabled
	})

	ginkgo.It("should support seccomp alpha runtime/default annotation [Feature:Seccomp] [LinuxOnly]", func() {
		// TODO: port to SecurityContext as soon as seccomp is out of alpha
		pod := scTestPod(false, false)
		pod.Annotations[v1.SeccompContainerAnnotationKeyPrefix+"test-container"] = v1.SeccompProfileRuntimeDefault
		pod.Spec.Containers[0].Command = []string{"grep", "ecc", "/proc/self/status"}
		f.TestContainerOutput(v1.SeccompPodAnnotationKey, pod, 0, []string{"2"}) // seccomp filtered
	})

	ginkgo.It("should support seccomp default which is unconfined [Feature:Seccomp] [LinuxOnly]", func() {
		// TODO: port to SecurityContext as soon as seccomp is out of alpha
		pod := scTestPod(false, false)
		pod.Spec.Containers[0].Command = []string{"grep", "ecc", "/proc/self/status"}
		f.TestContainerOutput(v1.SeccompPodAnnotationKey, pod, 0, []string{"0"}) // seccomp disabled
	})
})

func testPodSELinuxLabeling(f *framework.Framework, hostIPC bool, hostPID bool) {
	// Write and read a file with an empty_dir volume
	// with a pod with the MCS label s0:c0,c1
	pod := scTestPod(hostIPC, hostPID)
	volumeName := "test-volume"
	mountPath := "/mounted_volume"
	pod.Spec.Containers[0].VolumeMounts = []v1.VolumeMount{
		{
			Name:      volumeName,
			MountPath: mountPath,
		},
	}
	pod.Spec.Volumes = []v1.Volume{
		{
			Name: volumeName,
			VolumeSource: v1.VolumeSource{
				EmptyDir: &v1.EmptyDirVolumeSource{
					Medium: v1.StorageMediumDefault,
				},
			},
		},
	}
	pod.Spec.SecurityContext.SELinuxOptions = &v1.SELinuxOptions{
		Level: "s0:c0,c1",
	}
	pod.Spec.Containers[0].Command = []string{"sleep", "6000"}

	client := f.ClientSet.CoreV1().Pods(f.Namespace.Name)
	pod, err := client.Create(context.TODO(), pod, metav1.CreateOptions{})

	framework.ExpectNoError(err, "Error creating pod %v", pod)
	framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(f.ClientSet, pod))

	testContent := "hello"
	testFilePath := mountPath + "/TEST"
	tk := e2ekubectl.NewTestKubeconfig(framework.TestContext.CertDir, framework.TestContext.Host, framework.TestContext.KubeConfig, framework.TestContext.KubeContext, framework.TestContext.KubectlPath, f.Namespace.Name)
	err = tk.WriteFileViaContainer(pod.Name, pod.Spec.Containers[0].Name, testFilePath, testContent)
	framework.ExpectNoError(err)
	content, err := tk.ReadFileViaContainer(pod.Name, pod.Spec.Containers[0].Name, testFilePath)
	framework.ExpectNoError(err)
	gomega.Expect(content).To(gomega.ContainSubstring(testContent))

	foundPod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(context.TODO(), pod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err)

	// Confirm that the file can be accessed from a second
	// pod using host_path with the same MCS label
	volumeHostPath := fmt.Sprintf("%s/pods/%s/volumes/kubernetes.io~empty-dir/%s", framework.TestContext.KubeVolumeDir, foundPod.UID, volumeName)
	ginkgo.By(fmt.Sprintf("confirming a container with the same label can read the file under --volume-dir=%s", framework.TestContext.KubeVolumeDir))
	pod = scTestPod(hostIPC, hostPID)
	pod.Spec.NodeName = foundPod.Spec.NodeName
	volumeMounts := []v1.VolumeMount{
		{
			Name:      volumeName,
			MountPath: mountPath,
		},
	}
	volumes := []v1.Volume{
		{
			Name: volumeName,
			VolumeSource: v1.VolumeSource{
				HostPath: &v1.HostPathVolumeSource{
					Path: volumeHostPath,
				},
			},
		},
	}
	pod.Spec.Containers[0].VolumeMounts = volumeMounts
	pod.Spec.Volumes = volumes
	pod.Spec.Containers[0].Command = []string{"cat", testFilePath}
	pod.Spec.SecurityContext.SELinuxOptions = &v1.SELinuxOptions{
		Level: "s0:c0,c1",
	}
	f.TestContainerOutput("Pod with same MCS label reading test file", pod, 0, []string{testContent})

	// Confirm that the same pod with a different MCS
	// label cannot access the volume
	ginkgo.By("confirming a container with a different MCS label is unable to read the file")
	pod = scTestPod(hostIPC, hostPID)
	pod.Spec.Volumes = volumes
	pod.Spec.Containers[0].VolumeMounts = volumeMounts
	pod.Spec.Containers[0].Command = []string{"sleep", "6000"}
	pod.Spec.SecurityContext.SELinuxOptions = &v1.SELinuxOptions{
		Level: "s0:c2,c3",
	}
	_, err = client.Create(context.TODO(), pod, metav1.CreateOptions{})
	framework.ExpectNoError(err, "Error creating pod %v", pod)

	err = e2epod.WaitForPodNameRunningInNamespace(f.ClientSet, pod.Name, f.Namespace.Name)
	framework.ExpectNoError(err, "Error waiting for pod to run %v", pod)

	// for this to work, SELinux should be in enforcing mode, so let's check that
	isEnforced, err := tk.ReadFileViaContainer(pod.Name, "test-container", "/sys/fs/selinux/enforce")
	if err == nil && isEnforced == "1" {
		_, err = tk.ReadFileViaContainer(pod.Name, "test-container", testFilePath)
		framework.ExpectError(err, "expecting SELinux to not let the container with different MCS label to read the file")
	}
}
