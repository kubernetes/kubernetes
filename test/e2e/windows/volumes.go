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

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

const (
	emptyDirVolumePath = "C:\\test-volume"
	hostMapPath        = "C:\\tmp"
	containerName      = "test-container"
	volumeName         = "test-volume"
)

var _ = sigDescribe(feature.Windows, "Windows volume mounts", skipUnlessWindows(func() {
	f := framework.NewDefaultFramework("windows-volumes")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
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

		ginkgo.It("container should have readOnly permissions on emptyDir", func(ctx context.Context) {

			ginkgo.By("creating a container with readOnly permissions on emptyDir volume")
			doReadOnlyTest(ctx, f, emptyDirSource, emptyDirVolumePath)

			ginkgo.By("creating two containers, one with readOnly permissions the other with read-write permissions on emptyDir volume")
			doReadWriteReadOnlyTest(ctx, f, emptyDirSource, emptyDirVolumePath)
		})

		ginkgo.It("container should have readOnly permissions on hostMapPath", func(ctx context.Context) {

			ginkgo.By("creating a container with readOnly permissions on hostMap volume")
			doReadOnlyTest(ctx, f, hostMapSource, hostMapPath)

			ginkgo.By("creating two containers, one with readOnly permissions the other with read-write permissions on hostMap volume")
			doReadWriteReadOnlyTest(ctx, f, hostMapSource, hostMapPath)
		})

	})
}))

func doReadOnlyTest(ctx context.Context, f *framework.Framework, source v1.VolumeSource, volumePath string) {
	var (
		filePath = volumePath + "\\test-file.txt"
		podName  = "pod-" + string(uuid.NewUUID())
		pod      = testPodWithROVolume(podName, source, volumePath)
	)
	pod.Spec.NodeSelector = map[string]string{
		"kubernetes.io/os": "windows",
	}

	pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)
	ginkgo.By("verifying that pod has the correct nodeSelector")
	gomega.Expect(pod.Spec.NodeSelector).To(gomega.HaveKeyWithValue("kubernetes.io/os", "windows"), "pod.spec.nodeSelector")

	cmd := []string{"cmd", "/c", "echo windows-volume-test", ">", filePath}

	ginkgo.By("verifying that pod will get an error when writing to a volume that is readonly")
	_, stderr, _ := e2epod.ExecCommandInContainerWithFullOutput(f, podName, containerName, cmd...)
	gomega.Expect(stderr).To(gomega.Equal("Access is denied."))
}

func doReadWriteReadOnlyTest(ctx context.Context, f *framework.Framework, source v1.VolumeSource, volumePath string) {
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
		Image: imageutils.GetE2EImage(imageutils.Pause),
		VolumeMounts: []v1.VolumeMount{
			{
				Name:      volumeName,
				MountPath: volumePath,
			},
		},
	}

	pod.Spec.Containers = append(pod.Spec.Containers, rwcontainer)
	pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)

	ginkgo.By("verifying that pod has the correct nodeSelector")
	gomega.Expect(pod.Spec.NodeSelector).To(gomega.HaveKeyWithValue("kubernetes.io/os", "windows"), "pod.spec.nodeSelector")

	ginkgo.By("verifying that pod can write to a volume with read/write access")
	writecmd := []string{"cmd", "/c", "echo windows-volume-test", ">", filePath}
	stdoutRW, stderrRW, errRW := e2epod.ExecCommandInContainerWithFullOutput(f, podName, rwcontainerName, writecmd...)
	msg := fmt.Sprintf("cmd: %v, stdout: %q, stderr: %q", writecmd, stdoutRW, stderrRW)
	framework.ExpectNoError(errRW, msg)

	ginkgo.By("verifying that pod will get an error when writing to a volume that is readonly")
	_, stderr, _ := e2epod.ExecCommandInContainerWithFullOutput(f, podName, containerName, writecmd...)
	gomega.Expect(stderr).To(gomega.Equal("Access is denied."))

	ginkgo.By("verifying that pod can read from a volume that is readonly")
	readcmd := []string{"cmd", "/c", "type", filePath}
	readout, readerr, err := e2epod.ExecCommandInContainerWithFullOutput(f, podName, containerName, readcmd...)
	readmsg := fmt.Sprintf("cmd: %v, stdout: %q, stderr: %q", readcmd, readout, readerr)
	gomega.Expect(readout).To(gomega.Equal("windows-volume-test"))
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
					Image: imageutils.GetE2EImage(imageutils.Pause),
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
