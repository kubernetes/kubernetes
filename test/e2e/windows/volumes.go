/*
Copyright 2018 The Kubernetes Authors.

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
	"fmt"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	semver "github.com/blang/semver/v4"
	"github.com/onsi/ginkgo"
)

const (
	emptyDirVolumePath = "C:\\test-volume"
	hostMapPath        = "C:\\tmp"
	containerName      = "test-container"
	volumeName         = "test-volume"
)

var (
	image           = imageutils.GetE2EImage(imageutils.Pause)
	powershellImage = imageutils.GetE2EImage(imageutils.BusyBox)
)

var _ = SIGDescribe("[Feature:Windows] Windows volume mounts ", func() {
	f := framework.NewDefaultFramework("windows-volumes")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged
	var (
		emptyDirSource = v1.VolumeSource{
			EmptyDir: &v1.EmptyDirVolumeSource{
				Medium: v1.StorageMediumDefault,
			},
		}
		hostPathDirectoryOrCreate = v1.HostPathDirectoryOrCreate
		hostMapSource             = v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{
				Path: hostMapPath,
				Type: &hostPathDirectoryOrCreate,
			},
		}
	)
	ginkgo.BeforeEach(func() {
		e2eskipper.SkipUnlessNodeOSDistroIs("windows")
	})

	ginkgo.Context("check volume mount permissions", func() {

		ginkgo.It("container should have readOnly permissions on emptyDir", func() {

			ginkgo.By("creating a container with readOnly permissions on emptyDir volume")
			doReadOnlyTest(f, emptyDirSource, emptyDirVolumePath)

			ginkgo.By("creating two containers, one with readOnly permissions the other with read-write permissions on emptyDir volume")
			doReadWriteReadOnlyTest(f, emptyDirSource, emptyDirVolumePath)
		})

		ginkgo.It("container should have readOnly permissions on hostMapPath", func() {

			ginkgo.By("creating a container with readOnly permissions on hostMap volume")
			doReadOnlyTest(f, hostMapSource, hostMapPath)

			ginkgo.By("creating two containers, one with readOnly permissions the other with read-write permissions on hostMap volume")
			doReadWriteReadOnlyTest(f, hostMapSource, hostMapPath)
		})
	})

	ginkgo.It("validate rootfs size can be set larger than 20Gb", func() {
		doSetRootFSSizeTest(f)
	})
})

func doReadOnlyTest(f *framework.Framework, source v1.VolumeSource, volumePath string) {
	var (
		filePath = volumePath + "\\test-file.txt"
		podName  = "pod-" + string(uuid.NewUUID())
		pod      = testPodWithROVolume(podName, source, volumePath)
	)
	pod.Spec.NodeSelector = map[string]string{
		"kubernetes.io/os": "windows",
	}

	pod = f.PodClient().CreateSync(pod)
	ginkgo.By("verifying that pod has the correct nodeSelector")
	framework.ExpectEqual(pod.Spec.NodeSelector["kubernetes.io/os"], "windows")

	cmd := []string{"cmd", "/c", "echo windows-volume-test", ">", filePath}

	ginkgo.By("verifying that pod will get an error when writing to a volume that is readonly")
	_, stderr, _ := f.ExecCommandInContainerWithFullOutput(podName, containerName, cmd...)
	framework.ExpectEqual(stderr, "Access is denied.")
}

func doReadWriteReadOnlyTest(f *framework.Framework, source v1.VolumeSource, volumePath string) {
	var (
		filePath        = volumePath + "\\test-file" + string(uuid.NewUUID())
		podName         = "pod-" + string(uuid.NewUUID())
		pod             = testPodWithROVolume(podName, source, volumePath)
		rwcontainerName = containerName + "-rw"
	)
	pod.Spec.NodeSelector = map[string]string{
		"kubernetes.io/os": "windows",
	}

	rwcontainer := v1.Container{
		Name:  containerName + "-rw",
		Image: image,
		VolumeMounts: []v1.VolumeMount{
			{
				Name:      volumeName,
				MountPath: volumePath,
			},
		},
	}

	pod.Spec.Containers = append(pod.Spec.Containers, rwcontainer)
	pod = f.PodClient().CreateSync(pod)

	ginkgo.By("verifying that pod has the correct nodeSelector")
	framework.ExpectEqual(pod.Spec.NodeSelector["kubernetes.io/os"], "windows")

	ginkgo.By("verifying that pod can write to a volume with read/write access")
	writecmd := []string{"cmd", "/c", "echo windows-volume-test", ">", filePath}
	stdoutRW, stderrRW, errRW := f.ExecCommandInContainerWithFullOutput(podName, rwcontainerName, writecmd...)
	msg := fmt.Sprintf("cmd: %v, stdout: %q, stderr: %q", writecmd, stdoutRW, stderrRW)
	framework.ExpectNoError(errRW, msg)

	ginkgo.By("verifying that pod will get an error when writing to a volume that is readonly")
	_, stderr, _ := f.ExecCommandInContainerWithFullOutput(podName, containerName, writecmd...)
	framework.ExpectEqual(stderr, "Access is denied.")

	ginkgo.By("verifying that pod can read from a volume that is readonly")
	readcmd := []string{"cmd", "/c", "type", filePath}
	readout, readerr, err := f.ExecCommandInContainerWithFullOutput(podName, containerName, readcmd...)
	readmsg := fmt.Sprintf("cmd: %v, stdout: %q, stderr: %q", readcmd, readout, readerr)
	framework.ExpectEqual(readout, "windows-volume-test")
	framework.ExpectNoError(err, readmsg)
}

// testPodWithROVolume makes a minimal pod defining a volume input source. Similarly to
// other tests for sig-windows this should append a nodeSelector for windows.
func testPodWithROVolume(podName string, source v1.VolumeSource, path string) *v1.Pod {
	return &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  containerName,
					Image: image,
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      volumeName,
							MountPath: path,
							ReadOnly:  true,
						},
					},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
			Volumes: []v1.Volume{
				{
					Name:         volumeName,
					VolumeSource: source,
				},
			},
		},
	}
}

func getNodeContainerRuntimeAndVersion(n v1.Node) (string, semver.Version, error) {
	containerRuntimeVersionString := n.Status.NodeInfo.DeepCopy().ContainerRuntimeVersion
	parts := strings.Split(containerRuntimeVersionString, "://")

	if len(parts) != 2 {
		return "", semver.Version{}, fmt.Errorf("Could not get container runtime and version from '%s'", containerRuntimeVersionString)
	}

	v, err := semver.ParseTolerant(parts[1])
	if err != nil {
		return "", semver.Version{}, err
	}

	return parts[0], v, nil
}

func newRootFSSizeTestPod(nodeName, podName string) *v1.Pod {
	return &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "rootfs-size-test",
					Image: powershellImage,
					Command: []string{
						"powershell.exe",
						"-Command",
						"if (-not ((Get-PsDrive -Name C).Free -gt 21474836480)) { exit 1 } else { exit 0 }",
					},
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceEphemeralStorage: resource.MustParse("30Gi"),
						},
					},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
			NodeName:      nodeName,
		},
	}
}

// The default size of the volume rootfs volume for Windows containers is 20Gb.
// In containerd v1.7+ this value can be specified via tha CRI fields added with
// https://github.com/kubernetes/kubernetes/pull/108894.
// this test validates this behavior.
func doSetRootFSSizeTest(f *framework.Framework) {
	ginkgo.Describe("verify container rootfs volume size can be specified", func() {
		ginkgo.By("Selecting a Windows node")
		targetNode, err := findWindowsNode(f)
		framework.ExpectNoError(err, "Error finding Windows node")
		framework.Logf("Using node: %v", targetNode.Name)

		ginkgo.By("Ensuring node is running containerd v1.7+")
		r, v, err := getNodeContainerRuntimeAndVersion(targetNode)
		framework.ExpectNoError(err, "Error getting node container runtime and version")
		framework.Logf("Got runtime: %s, version %v", r, v)

		if !strings.EqualFold(r, "containerd") {
			e2eskipper.Skipf("container runtime is not containerd")
		}

		v1dot7 := semver.MustParse("1.7.0")
		if v.LT(v1dot7) {
			e2eskipper.Skipf("container runtime version less than v1.7")
		}

		ginkgo.By("Scheudling a pod with a 30Gi empheral-storage limit")
		podName := "rootfs-test-pod"
		f.PodClient().Create(newRootFSSizeTestPod(targetNode.Name, podName))

		ginkgo.By("Waiting for pod to run")
		f.PodClient().WaitForFinish(podName, 3*time.Minute)

		ginkgo.By("Then ensuring pod finished running successfully")
		p, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(
			context.TODO(),
			podName,
			metav1.GetOptions{})

		framework.ExpectNoError(err, "Error retrieving pod")
		framework.ExpectEqual(p.Status.Phase, v1.PodSucceeded)
	})
}
