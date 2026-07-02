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
	"strings"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ekubectl "k8s.io/kubernetes/test/e2e/framework/kubectl"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eoutput "k8s.io/kubernetes/test/e2e/framework/pod/output"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

// SeccompProcStatusField is the field of /proc/$PID/status referencing the seccomp filter type.
const SeccompProcStatusField = "Seccomp:"

// ProcSelfStatusPath is the path to /proc/self/status.
const ProcSelfStatusPath = "/proc/self/status"

const selinuxTestMountPath = "/mounted_volume"

var selinuxKeepAliveCommand = []string{"sleep", "infinity"}

var supportedSELinuxNodeOSDistros = []string{"debian", "ubuntu", "custom"}

type selinuxVolumeFileSpec struct {
	subpath string
	content string
	write   bool
}

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
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.It("should support pod.Spec.SecurityContext.SupplementalGroups [LinuxOnly]", func(ctx context.Context) {
		pod := scTestPod(false, false)
		pod.Spec.Containers[0].Command = []string{"id", "-G"}
		pod.Spec.SecurityContext.SupplementalGroups = []int64{1234, 5678}
		groups := []string{"1234", "5678"}
		e2eoutput.TestContainerOutput(ctx, f, "pod.Spec.SecurityContext.SupplementalGroups", pod, 0, groups)
	})

	ginkgo.When("if the container's primary UID belongs to some groups in the image [LinuxOnly]", func() {
		ginkgo.It("should add pod.Spec.SecurityContext.SupplementalGroups to them [LinuxOnly] in resultant supplementary groups for the container processes", func(ctx context.Context) {
			uidInImage := int64(1000)
			gidDefinedInImage := int64(50000)
			supplementalGroup := int64(60000)
			agnhost := imageutils.GetConfig(imageutils.Agnhost)
			pod := scTestPod(false, false)
			pod.Spec.Containers[0].Image = agnhost.GetE2EImage()
			pod.Spec.Containers[0].Command = []string{"id", "-G"}
			pod.Spec.SecurityContext.SupplementalGroups = []int64{int64(supplementalGroup)}
			pod.Spec.SecurityContext.RunAsUser = &uidInImage

			// In specified image(agnhost E2E image),
			// - user-defined-in-image(uid=1000) is defined
			// - user-defined-in-image belongs to group-defined-in-image(gid=50000)
			// thus, resultant supplementary group of the container processes should be
			// - 1000: self
			// - 50000: pre-defined groups define in the container image of self(uid=1000)
			// - 60000: SupplementalGroups
			// $ id -G
			// 1000 50000 60000
			e2eoutput.TestContainerOutput(
				ctx,
				f,
				"pod.Spec.SecurityContext.SupplementalGroups with pre-defined-group in the image",
				pod, 0,
				[]string{fmt.Sprintf("%d %d %d", uidInImage, gidDefinedInImage, supplementalGroup)},
			)
		})
	})

	ginkgo.It("should support pod.Spec.SecurityContext.RunAsUser [LinuxOnly]", func(ctx context.Context) {
		pod := scTestPod(false, false)
		userID := int64(1001)
		pod.Spec.SecurityContext.RunAsUser = &userID
		pod.Spec.Containers[0].Command = []string{"sh", "-c", "id"}

		e2eoutput.TestContainerOutput(ctx, f, "pod.Spec.SecurityContext.RunAsUser", pod, 0, []string{
			fmt.Sprintf("uid=%v", userID),
			fmt.Sprintf("gid=%v", 0),
		})
	})

	/*
		Release: v1.21
		Testname: Security Context, test RunAsGroup at pod level
		Description: Container is created with runAsUser and runAsGroup option by passing uid 1001 and gid 2002 at pod level. Pod MUST be in Succeeded phase.
		[LinuxOnly]: This test is marked as LinuxOnly since Windows does not support running as UID / GID.
	*/
	framework.ConformanceIt("should support pod.Spec.SecurityContext.RunAsUser And pod.Spec.SecurityContext.RunAsGroup [LinuxOnly]", func(ctx context.Context) {
		pod := scTestPod(false, false)
		userID := int64(1001)
		groupID := int64(2002)
		pod.Spec.SecurityContext.RunAsUser = &userID
		pod.Spec.SecurityContext.RunAsGroup = &groupID
		pod.Spec.Containers[0].Command = []string{"sh", "-c", "id"}

		e2eoutput.TestContainerOutput(ctx, f, "pod.Spec.SecurityContext.RunAsUser", pod, 0, []string{
			fmt.Sprintf("uid=%v", userID),
			fmt.Sprintf("gid=%v", groupID),
		})
	})

	ginkgo.It("should support container.SecurityContext.RunAsUser [LinuxOnly]", func(ctx context.Context) {
		pod := scTestPod(false, false)
		userID := int64(1001)
		overrideUserID := int64(1002)
		pod.Spec.SecurityContext.RunAsUser = &userID
		pod.Spec.Containers[0].SecurityContext = new(v1.SecurityContext)
		pod.Spec.Containers[0].SecurityContext.RunAsUser = &overrideUserID
		pod.Spec.Containers[0].Command = []string{"sh", "-c", "id"}

		e2eoutput.TestContainerOutput(ctx, f, "pod.Spec.SecurityContext.RunAsUser", pod, 0, []string{
			fmt.Sprintf("uid=%v", overrideUserID),
			fmt.Sprintf("gid=%v", 0),
		})
	})

	/*
		Release: v1.21
		Testname: Security Context, test RunAsGroup at container level
		Description: Container is created with runAsUser and runAsGroup option by passing uid 1001 and gid 2002 at containr level. Pod MUST be in Succeeded phase.
		[LinuxOnly]: This test is marked as LinuxOnly since Windows does not support running as UID / GID.
	*/
	framework.ConformanceIt("should support container.SecurityContext.RunAsUser And container.SecurityContext.RunAsGroup [LinuxOnly]", func(ctx context.Context) {
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

		e2eoutput.TestContainerOutput(ctx, f, "pod.Spec.SecurityContext.RunAsUser", pod, 0, []string{
			fmt.Sprintf("uid=%v", overrideUserID),
			fmt.Sprintf("gid=%v", overrideGroupID),
		})
	})

	f.Context("volume SELinux relabeling [LinuxOnly]", feature.SELinux, func() {
		f.It("should support volume SELinux relabeling", f.WithLabel("LinuxOnly"), func(ctx context.Context) {
			testPodSELinuxLabeling(ctx, f, false, false, "kubernetes.io~empty-dir", emptyDirVolume("test-volume"), selinuxVolumeFileSpec{subpath: "TEST", content: "hello", write: true})
		})

		f.It("should support volume SELinux relabeling when using hostIPC", f.WithLabel("LinuxOnly"), func(ctx context.Context) {
			testPodSELinuxLabeling(ctx, f, true, false, "kubernetes.io~empty-dir", emptyDirVolume("test-volume"), selinuxVolumeFileSpec{subpath: "TEST", content: "hello", write: true})
		})

		f.It("should support volume SELinux relabeling when using hostPID", f.WithLabel("LinuxOnly"), func(ctx context.Context) {
			testPodSELinuxLabeling(ctx, f, false, true, "kubernetes.io~empty-dir", emptyDirVolume("test-volume"), selinuxVolumeFileSpec{subpath: "TEST", content: "hello", write: true})
		})

		f.It("should support configmap volume SELinux relabeling", f.WithLabel("LinuxOnly"), func(ctx context.Context) {
			configMap := createSELinuxRelabelConfigMap(ctx, f)
			testPodSELinuxLabeling(ctx, f, false, false, "kubernetes.io~configmap", configMapVolume(configMap.Name), selinuxVolumeFileSpec{subpath: "initial", content: "data", write: false})
		})

		f.It("should support secret volume SELinux relabeling", f.WithLabel("LinuxOnly"), func(ctx context.Context) {
			secret := createSELinuxRelabelSecret(ctx, f)
			testPodSELinuxLabeling(ctx, f, false, false, "kubernetes.io~secret", secretVolume(secret.Name), selinuxVolumeFileSpec{subpath: "initial", content: "data", write: false})
		})
	})

	ginkgo.It("should support seccomp unconfined on the container [LinuxOnly]", func(ctx context.Context) {
		pod := scTestPod(false, false)
		pod.Spec.Containers[0].SecurityContext = &v1.SecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeUnconfined}}
		pod.Spec.SecurityContext = &v1.PodSecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeRuntimeDefault}}
		pod.Spec.Containers[0].Command = []string{"grep", SeccompProcStatusField, ProcSelfStatusPath}
		e2eoutput.TestContainerOutput(ctx, f, "seccomp unconfined container", pod, 0, []string{"0"}) // seccomp disabled
	})

	ginkgo.It("should support seccomp unconfined on the pod [LinuxOnly]", func(ctx context.Context) {
		pod := scTestPod(false, false)
		pod.Spec.SecurityContext = &v1.PodSecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeUnconfined}}
		pod.Spec.Containers[0].Command = []string{"grep", SeccompProcStatusField, ProcSelfStatusPath}
		e2eoutput.TestContainerOutput(ctx, f, "seccomp unconfined pod", pod, 0, []string{"0"}) // seccomp disabled
	})

	ginkgo.It("should support seccomp runtime/default [LinuxOnly]", func(ctx context.Context) {
		pod := scTestPod(false, false)
		pod.Spec.Containers[0].SecurityContext = &v1.SecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeRuntimeDefault}}
		pod.Spec.Containers[0].Command = []string{"grep", SeccompProcStatusField, ProcSelfStatusPath}
		e2eoutput.TestContainerOutput(ctx, f, "seccomp runtime/default", pod, 0, []string{"2"}) // seccomp filtered
	})
})

func testPodSELinuxLabeling(ctx context.Context, f *framework.Framework, hostIPC bool, hostPID bool, volumePlugin string, volumes []v1.Volume, fileSpec selinuxVolumeFileSpec) {
	if !isSupportedSELinuxDistro() {
		e2eskipper.Skipf("SELinux volume relabeling tests are supported only on %+v", supportedSELinuxNodeOSDistros)
	}

	volumeName := volumes[0].Name
	mountPath := selinuxTestMountPath
	testFilePath := mountPath + "/" + fileSpec.subpath

	// Write and read a file with a pod that has the MCS label s0:c0,c1.
	pod := scTestPod(hostIPC, hostPID)
	pod.Spec.Containers[0].VolumeMounts = []v1.VolumeMount{{
		Name:      volumeName,
		MountPath: mountPath,
	}}
	pod.Spec.Volumes = volumes
	pod.Spec.SecurityContext.SELinuxOptions = &v1.SELinuxOptions{
		Level: "s0:c0,c1",
	}
	pod.Spec.Containers[0].Command = selinuxKeepAliveCommand

	client := f.ClientSet.CoreV1().Pods(f.Namespace.Name)
	pod, err := client.Create(ctx, pod, metav1.CreateOptions{})
	framework.ExpectNoError(err, "Error creating pod %v", pod)
	framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod))

	tk := e2ekubectl.NewTestKubeconfig(framework.TestContext.CertDir, framework.TestContext.Host, framework.TestContext.KubeConfig, framework.TestContext.KubeContext, framework.TestContext.KubectlPath, f.Namespace.Name)
	requireSELinuxEnforcing(tk, pod.Name, pod.Spec.Containers[0].Name)

	if fileSpec.write {
		err = tk.WriteFileViaContainer(pod.Name, pod.Spec.Containers[0].Name, testFilePath, fileSpec.content)
		framework.ExpectNoError(err)
	}
	content, err := tk.ReadFileViaContainer(pod.Name, pod.Spec.Containers[0].Name, testFilePath)
	framework.ExpectNoError(err)
	gomega.Expect(content).To(gomega.ContainSubstring(fileSpec.content))

	foundPod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err)

	// Confirm that the file can be accessed from a second pod using host_path with the same MCS label.
	volumeHostPath := fmt.Sprintf("%s/pods/%s/volumes/%s/%s", framework.TestContext.KubeletRootDir, foundPod.UID, volumePlugin, volumeName)
	ginkgo.By(fmt.Sprintf("confirming a container with the same label can read the file under --kubelet-root-dir=%s", framework.TestContext.KubeletRootDir))
	pod = scTestPod(hostIPC, hostPID)
	pod.Spec.NodeName = foundPod.Spec.NodeName
	volumeMounts := []v1.VolumeMount{{
		Name:      volumeName,
		MountPath: mountPath,
	}}
	hostPathVolumes := []v1.Volume{{
		Name: volumeName,
		VolumeSource: v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{
				Path: volumeHostPath,
			},
		},
	}}
	pod.Spec.Containers[0].VolumeMounts = volumeMounts
	pod.Spec.Volumes = hostPathVolumes
	pod.Spec.Containers[0].Command = []string{"cat", testFilePath}
	pod.Spec.SecurityContext.SELinuxOptions = &v1.SELinuxOptions{
		Level: "s0:c0,c1",
	}
	e2eoutput.TestContainerOutput(ctx, f, "Pod with same MCS label reading test file", pod, 0, []string{fileSpec.content})

	// Confirm that a pod with a different MCS label cannot access the volume.
	ginkgo.By("confirming a container with a different MCS label is unable to read the file")
	pod = scTestPod(hostIPC, hostPID)
	pod.Spec.NodeName = foundPod.Spec.NodeName
	pod.Spec.Volumes = hostPathVolumes
	pod.Spec.Containers[0].VolumeMounts = volumeMounts
	pod.Spec.Containers[0].Command = selinuxKeepAliveCommand
	pod.Spec.SecurityContext.SELinuxOptions = &v1.SELinuxOptions{
		Level: "s0:c2,c3",
	}
	pod, err = client.Create(ctx, pod, metav1.CreateOptions{})
	framework.ExpectNoError(err, "Error creating pod %v", pod)
	framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name))

	_, err = tk.ReadFileViaContainer(pod.Name, pod.Spec.Containers[0].Name, testFilePath)
	gomega.Expect(err).To(gomega.HaveOccurred(), "expecting SELinux to not let the container with different MCS label to read the file")
}

func isSupportedSELinuxDistro() bool {
	for _, distro := range supportedSELinuxNodeOSDistros {
		if framework.TestContext.NodeOSDistro == distro {
			return true
		}
	}
	return false
}

func requireSELinuxEnforcing(tk *e2ekubectl.TestKubeconfig, podName, containerName string) {
	isEnforced, err := tk.ReadFileViaContainer(podName, containerName, "/sys/fs/selinux/enforce")
	framework.ExpectNoError(err, "reading SELinux enforce status")
	gomega.Expect(strings.TrimSpace(isEnforced)).To(gomega.Equal("1"), "SELinux must be in enforcing mode for [Feature:SELinux] tests")
}

func emptyDirVolume(volumeName string) []v1.Volume {
	return []v1.Volume{{
		Name: volumeName,
		VolumeSource: v1.VolumeSource{
			EmptyDir: &v1.EmptyDirVolumeSource{
				Medium: v1.StorageMediumDefault,
			},
		},
	}}
}

func configMapVolume(configMapName string) []v1.Volume {
	return []v1.Volume{{
		Name: "test-volume",
		VolumeSource: v1.VolumeSource{
			ConfigMap: &v1.ConfigMapVolumeSource{
				LocalObjectReference: v1.LocalObjectReference{Name: configMapName},
			},
		},
	}}
}

func secretVolume(secretName string) []v1.Volume {
	return []v1.Volume{{
		Name: "test-volume",
		VolumeSource: v1.VolumeSource{
			Secret: &v1.SecretVolumeSource{
				SecretName: secretName,
			},
		},
	}}
}

func createSELinuxRelabelConfigMap(ctx context.Context, f *framework.Framework) *v1.ConfigMap {
	configMap, err := f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(ctx, &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name: "selinux-relabel-configmap-" + string(uuid.NewUUID()),
		},
		Data: map[string]string{
			"initial": "data",
		},
	}, metav1.CreateOptions{})
	framework.ExpectNoError(err)
	return configMap
}

func createSELinuxRelabelSecret(ctx context.Context, f *framework.Framework) *v1.Secret {
	secret, err := f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Create(ctx, &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name: "selinux-relabel-secret-" + string(uuid.NewUUID()),
		},
		StringData: map[string]string{
			"initial": "data",
		},
	}, metav1.CreateOptions{})
	framework.ExpectNoError(err)
	return secret
}
