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
	"fmt"
	"path"
	"time"

	. "github.com/onsi/ginkgo"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

const (
	volumePath = "/test-volume"
)

var (
	testImageRootUid    = imageutils.GetE2EImage(imageutils.Mounttest)
	testImageNonRootUid = imageutils.GetE2EImage(imageutils.MounttestUser)
)

var _ = Describe("[sig-storage] EmptyDir volumes", func() {
	f := framework.NewDefaultFramework("emptydir")

	Context("when FSGroup is specified [NodeFeature:FSGroup]", func() {
		It("new files should be created with FSGroup ownership when container is root", func() {
			doTestSetgidFSGroup(f, testImageRootUid, v1.StorageMediumMemory)
		})

		It("new files should be created with FSGroup ownership when container is non-root", func() {
			doTestSetgidFSGroup(f, testImageNonRootUid, v1.StorageMediumMemory)
		})

		It("nonexistent volume subPath should have the correct mode and owner using FSGroup", func() {
			doTestSubPathFSGroup(f, testImageNonRootUid, v1.StorageMediumMemory)
		})

		It("files with FSGroup ownership should support (root,0644,tmpfs)", func() {
			doTest0644FSGroup(f, testImageRootUid, v1.StorageMediumMemory)
		})

		It("volume on default medium should have the correct mode using FSGroup", func() {
			doTestVolumeModeFSGroup(f, testImageRootUid, v1.StorageMediumDefault)
		})

		It("volume on tmpfs should have the correct mode using FSGroup", func() {
			doTestVolumeModeFSGroup(f, testImageRootUid, v1.StorageMediumMemory)
		})
	})

	/*
		Release : v1.9
		Testname: EmptyDir, medium memory, volume mode default
		Description: A Pod created with an 'emptyDir' Volume and 'medium' as 'Memory', the volume MUST have mode set as -rwxrwxrwx and mount type set to tmpfs.
		This test is marked LinuxOnly since Windows does not support setting specific file permissions, or the medium = 'Memory'.
	*/
	framework.ConformanceIt("volume on tmpfs should have the correct mode [LinuxOnly] [NodeConformance]", func() {
		doTestVolumeMode(f, testImageRootUid, v1.StorageMediumMemory)
	})

	/*
		Release : v1.9
		Testname: EmptyDir, medium memory, volume mode 0644
		Description: A Pod created with an 'emptyDir' Volume and 'medium' as 'Memory', the volume mode set to 0644. The volume MUST have mode -rw-r--r-- and mount type set to tmpfs and the contents MUST be readable.
		This test is marked LinuxOnly since Windows does not support setting specific file permissions, or running as UID / GID, or the medium = 'Memory'.
	*/
	framework.ConformanceIt("should support (root,0644,tmpfs) [LinuxOnly] [NodeConformance]", func() {
		doTest0644(f, testImageRootUid, v1.StorageMediumMemory)
	})

	/*
		Release : v1.9
		Testname: EmptyDir, medium memory, volume mode 0666
		Description: A Pod created with an 'emptyDir' Volume and 'medium' as 'Memory', the volume mode set to 0666. The volume MUST have mode -rw-rw-rw- and mount type set to tmpfs and the contents MUST be readable.
		This test is marked LinuxOnly since Windows does not support setting specific file permissions, or running as UID / GID, or the medium = 'Memory'.
	*/
	framework.ConformanceIt("should support (root,0666,tmpfs) [LinuxOnly] [NodeConformance]", func() {
		doTest0666(f, testImageRootUid, v1.StorageMediumMemory)
	})

	/*
		Release : v1.9
		Testname: EmptyDir, medium memory, volume mode 0777
		Description: A Pod created with an 'emptyDir' Volume and 'medium' as 'Memory', the volume mode set to 0777.  The volume MUST have mode set as -rwxrwxrwx and mount type set to tmpfs and the contents MUST be readable.
		This test is marked LinuxOnly since Windows does not support setting specific file permissions, or running as UID / GID, or the medium = 'Memory'.
	*/
	framework.ConformanceIt("should support (root,0777,tmpfs) [LinuxOnly] [NodeConformance]", func() {
		doTest0777(f, testImageRootUid, v1.StorageMediumMemory)
	})

	/*
		Release : v1.9
		Testname: EmptyDir, medium memory, volume mode 0644, non-root user
		Description: A Pod created with an 'emptyDir' Volume and 'medium' as 'Memory', the volume mode set to 0644. Volume is mounted into the container where container is run as a non-root user. The volume MUST have mode -rw-r--r-- and mount type set to tmpfs and the contents MUST be readable.
		This test is marked LinuxOnly since Windows does not support setting specific file permissions, or running as UID / GID, or the medium = 'Memory'.
	*/
	framework.ConformanceIt("should support (non-root,0644,tmpfs) [LinuxOnly] [NodeConformance]", func() {
		doTest0644(f, testImageNonRootUid, v1.StorageMediumMemory)
	})

	/*
		Release : v1.9
		Testname: EmptyDir, medium memory, volume mode 0666,, non-root user
		Description: A Pod created with an 'emptyDir' Volume and 'medium' as 'Memory', the volume mode set to 0666. Volume is mounted into the container where container is run as a non-root user. The volume MUST have mode -rw-rw-rw- and mount type set to tmpfs and the contents MUST be readable.
		This test is marked LinuxOnly since Windows does not support setting specific file permissions, or running as UID / GID, or the medium = 'Memory'.
	*/
	framework.ConformanceIt("should support (non-root,0666,tmpfs) [LinuxOnly] [NodeConformance]", func() {
		doTest0666(f, testImageNonRootUid, v1.StorageMediumMemory)
	})

	/*
		Release : v1.9
		Testname: EmptyDir, medium memory, volume mode 0777, non-root user
		Description: A Pod created with an 'emptyDir' Volume and 'medium' as 'Memory', the volume mode set to 0777. Volume is mounted into the container where container is run as a non-root user. The volume MUST have mode -rwxrwxrwx and mount type set to tmpfs and the contents MUST be readable.
		This test is marked LinuxOnly since Windows does not support setting specific file permissions, or running as UID / GID, or the medium = 'Memory'.
	*/
	framework.ConformanceIt("should support (non-root,0777,tmpfs) [LinuxOnly] [NodeConformance]", func() {
		doTest0777(f, testImageNonRootUid, v1.StorageMediumMemory)
	})

	/*
		Release : v1.9
		Testname: EmptyDir, medium default, volume mode default
		Description: A Pod created with an 'emptyDir' Volume, the volume MUST have mode set as -rwxrwxrwx and mount type set to tmpfs.
		This test is marked LinuxOnly since Windows does not support setting specific file permissions.
	*/
	framework.ConformanceIt("volume on default medium should have the correct mode [LinuxOnly] [NodeConformance]", func() {
		doTestVolumeMode(f, testImageRootUid, v1.StorageMediumDefault)
	})

	/*
		Release : v1.9
		Testname: EmptyDir, medium default, volume mode 0644
		Description: A Pod created with an 'emptyDir' Volume, the volume mode set to 0644. The volume MUST have mode -rw-r--r-- and mount type set to tmpfs and the contents MUST be readable.
		This test is marked LinuxOnly since Windows does not support setting specific file permissions, or running as UID / GID.
	*/
	framework.ConformanceIt("should support (root,0644,default) [LinuxOnly] [NodeConformance]", func() {
		doTest0644(f, testImageRootUid, v1.StorageMediumDefault)
	})

	/*
		Release : v1.9
		Testname: EmptyDir, medium default, volume mode 0666
		Description: A Pod created with an 'emptyDir' Volume, the volume mode set to 0666. The volume MUST have mode -rw-rw-rw- and mount type set to tmpfs and the contents MUST be readable.
		This test is marked LinuxOnly since Windows does not support setting specific file permissions, or running as UID / GID.
	*/
	framework.ConformanceIt("should support (root,0666,default) [LinuxOnly] [NodeConformance]", func() {
		doTest0666(f, testImageRootUid, v1.StorageMediumDefault)
	})

	/*
		Release : v1.9
		Testname: EmptyDir, medium default, volume mode 0777
		Description: A Pod created with an 'emptyDir' Volume, the volume mode set to 0777.  The volume MUST have mode set as -rwxrwxrwx and mount type set to tmpfs and the contents MUST be readable.
		This test is marked LinuxOnly since Windows does not support setting specific file permissions, or running as UID / GID.
	*/
	framework.ConformanceIt("should support (root,0777,default) [LinuxOnly] [NodeConformance]", func() {
		doTest0777(f, testImageRootUid, v1.StorageMediumDefault)
	})

	/*
		Release : v1.9
		Testname: EmptyDir, medium default, volume mode 0644
		Description: A Pod created with an 'emptyDir' Volume, the volume mode set to 0644. Volume is mounted into the container where container is run as a non-root user. The volume MUST have mode -rw-r--r-- and mount type set to tmpfs and the contents MUST be readable.
		This test is marked LinuxOnly since Windows does not support setting specific file permissions, or running as UID / GID.
	*/
	framework.ConformanceIt("should support (non-root,0644,default) [LinuxOnly] [NodeConformance]", func() {
		doTest0644(f, testImageNonRootUid, v1.StorageMediumDefault)
	})

	/*
		Release : v1.9
		Testname: EmptyDir, medium default, volume mode 0666
		Description: A Pod created with an 'emptyDir' Volume, the volume mode set to 0666. Volume is mounted into the container where container is run as a non-root user. The volume MUST have mode -rw-rw-rw- and mount type set to tmpfs and the contents MUST be readable.
		This test is marked LinuxOnly since Windows does not support setting specific file permissions, or running as UID / GID.
	*/
	framework.ConformanceIt("should support (non-root,0666,default) [LinuxOnly] [NodeConformance]", func() {
		doTest0666(f, testImageNonRootUid, v1.StorageMediumDefault)
	})

	/*
		Release : v1.9
		Testname: EmptyDir, medium default, volume mode 0777
		Description: A Pod created with an 'emptyDir' Volume, the volume mode set to 0777. Volume is mounted into the container where container is run as a non-root user. The volume MUST have mode -rwxrwxrwx and mount type set to tmpfs and the contents MUST be readable.
		This test is marked LinuxOnly since Windows does not support setting specific file permissions, or running as UID / GID.
	*/
	framework.ConformanceIt("should support (non-root,0777,default) [LinuxOnly] [NodeConformance]", func() {
		doTest0777(f, testImageNonRootUid, v1.StorageMediumDefault)
	})

	It("pod should support shared volumes between containers", func() {
		var (
			volumeName                 = "shared-data"
			busyBoxMainVolumeMountPath = "/usr/share/volumeshare"
			busyBoxSubVolumeMountPath  = "/pod-data"
			busyBoxMainVolumeFilePath  = fmt.Sprintf("%s/shareddata.txt", busyBoxMainVolumeMountPath)
			busyBoxSubVolumeFilePath   = fmt.Sprintf("%s/shareddata.txt", busyBoxSubVolumeMountPath)
			message                    = "Hello from the busy-box sub-container"
			busyBoxMainContainerName   = "busybox-main-container"
			busyBoxSubContainerName    = "busybox-sub-container"
			resultString               = ""
			deletionGracePeriod        = int64(0)
		)

		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "pod-sharedvolume-" + string(uuid.NewUUID()),
			},
			Spec: v1.PodSpec{
				Volumes: []v1.Volume{
					{
						Name: volumeName,
						VolumeSource: v1.VolumeSource{
							EmptyDir: new(v1.EmptyDirVolumeSource),
						},
					},
				},
				Containers: []v1.Container{
					{
						Name:    busyBoxMainContainerName,
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"/bin/sh"},
						Args:    []string{"-c", "sleep 100000"},
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      volumeName,
								MountPath: busyBoxMainVolumeMountPath,
							},
						},
					},
					{
						Name:    busyBoxSubContainerName,
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"/bin/sh"},
						Args:    []string{"-c", fmt.Sprintf("echo %s > %s", message, busyBoxSubVolumeFilePath)},
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      volumeName,
								MountPath: busyBoxSubVolumeMountPath,
							},
						},
					},
				},
				TerminationGracePeriodSeconds: &deletionGracePeriod,
				RestartPolicy:                 v1.RestartPolicyNever,
			},
		}

		var err error
		By("Creating Pod")
		pod = f.PodClient().CreateSync(pod)

		By("Waiting for the pod running")
		err = f.WaitForPodRunning(pod.Name)
		framework.ExpectNoError(err, "failed to deploy pod %s", pod.Name)

		By("Geting the pod")
		pod, err = f.PodClient().Get(pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to get pod %s", pod.Name)

		By("Reading file content from the nginx-container")
		resultString, err = framework.LookForStringInFile(f.Namespace.Name, pod.Name, busyBoxMainContainerName, busyBoxMainVolumeFilePath, message, 30*time.Second)
		framework.ExpectNoError(err, "failed to match expected string %s with %s", message, resultString)
	})
})

const (
	containerName = "test-container"
	volumeName    = "test-volume"
)

func doTestSetgidFSGroup(f *framework.Framework, image string, medium v1.StorageMedium) {
	var (
		filePath = path.Join(volumePath, "test-file")
		source   = &v1.EmptyDirVolumeSource{Medium: medium}
		pod      = testPodWithVolume(testImageRootUid, volumePath, source)
	)

	pod.Spec.Containers[0].Args = []string{
		fmt.Sprintf("--fs_type=%v", volumePath),
		fmt.Sprintf("--new_file_0660=%v", filePath),
		fmt.Sprintf("--file_perm=%v", filePath),
		fmt.Sprintf("--file_owner=%v", filePath),
	}

	fsGroup := int64(123)
	pod.Spec.SecurityContext.FSGroup = &fsGroup

	msg := fmt.Sprintf("emptydir 0644 on %v", formatMedium(medium))
	out := []string{
		"perms of file \"/test-volume/test-file\": -rw-rw----",
		"content of file \"/test-volume/test-file\": mount-tester new file",
		"owner GID of \"/test-volume/test-file\": 123",
	}
	if medium == v1.StorageMediumMemory {
		out = append(out, "mount type of \"/test-volume\": tmpfs")
	}
	f.TestContainerOutput(msg, pod, 0, out)
}

func doTestSubPathFSGroup(f *framework.Framework, image string, medium v1.StorageMedium) {
	var (
		subPath = "test-sub"
		source  = &v1.EmptyDirVolumeSource{Medium: medium}
		pod     = testPodWithVolume(image, volumePath, source)
	)

	pod.Spec.Containers[0].Args = []string{
		fmt.Sprintf("--fs_type=%v", volumePath),
		fmt.Sprintf("--file_perm=%v", volumePath),
		fmt.Sprintf("--file_owner=%v", volumePath),
		fmt.Sprintf("--file_mode=%v", volumePath),
	}

	pod.Spec.Containers[0].VolumeMounts[0].SubPath = subPath

	fsGroup := int64(123)
	pod.Spec.SecurityContext.FSGroup = &fsGroup

	msg := fmt.Sprintf("emptydir subpath on %v", formatMedium(medium))
	out := []string{
		"perms of file \"/test-volume\": -rwxrwxrwx",
		"owner UID of \"/test-volume\": 0",
		"owner GID of \"/test-volume\": 123",
		"mode of file \"/test-volume\": dgtrwxrwxrwx",
	}
	if medium == v1.StorageMediumMemory {
		out = append(out, "mount type of \"/test-volume\": tmpfs")
	}
	f.TestContainerOutput(msg, pod, 0, out)
}

func doTestVolumeModeFSGroup(f *framework.Framework, image string, medium v1.StorageMedium) {
	var (
		source = &v1.EmptyDirVolumeSource{Medium: medium}
		pod    = testPodWithVolume(testImageRootUid, volumePath, source)
	)

	pod.Spec.Containers[0].Args = []string{
		fmt.Sprintf("--fs_type=%v", volumePath),
		fmt.Sprintf("--file_perm=%v", volumePath),
	}

	fsGroup := int64(1001)
	pod.Spec.SecurityContext.FSGroup = &fsGroup

	msg := fmt.Sprintf("emptydir volume type on %v", formatMedium(medium))
	out := []string{
		"perms of file \"/test-volume\": -rwxrwxrwx",
	}
	if medium == v1.StorageMediumMemory {
		out = append(out, "mount type of \"/test-volume\": tmpfs")
	}
	f.TestContainerOutput(msg, pod, 0, out)
}

func doTest0644FSGroup(f *framework.Framework, image string, medium v1.StorageMedium) {
	var (
		filePath = path.Join(volumePath, "test-file")
		source   = &v1.EmptyDirVolumeSource{Medium: medium}
		pod      = testPodWithVolume(image, volumePath, source)
	)

	pod.Spec.Containers[0].Args = []string{
		fmt.Sprintf("--fs_type=%v", volumePath),
		fmt.Sprintf("--new_file_0644=%v", filePath),
		fmt.Sprintf("--file_perm=%v", filePath),
	}

	fsGroup := int64(123)
	pod.Spec.SecurityContext.FSGroup = &fsGroup

	msg := fmt.Sprintf("emptydir 0644 on %v", formatMedium(medium))
	out := []string{
		"perms of file \"/test-volume/test-file\": -rw-r--r--",
		"content of file \"/test-volume/test-file\": mount-tester new file",
	}
	if medium == v1.StorageMediumMemory {
		out = append(out, "mount type of \"/test-volume\": tmpfs")
	}
	f.TestContainerOutput(msg, pod, 0, out)
}

func doTestVolumeMode(f *framework.Framework, image string, medium v1.StorageMedium) {
	var (
		source = &v1.EmptyDirVolumeSource{Medium: medium}
		pod    = testPodWithVolume(testImageRootUid, volumePath, source)
	)

	pod.Spec.Containers[0].Args = []string{
		fmt.Sprintf("--fs_type=%v", volumePath),
		fmt.Sprintf("--file_perm=%v", volumePath),
	}

	msg := fmt.Sprintf("emptydir volume type on %v", formatMedium(medium))
	out := []string{
		"perms of file \"/test-volume\": -rwxrwxrwx",
	}
	if medium == v1.StorageMediumMemory {
		out = append(out, "mount type of \"/test-volume\": tmpfs")
	}
	f.TestContainerOutput(msg, pod, 0, out)
}

func doTest0644(f *framework.Framework, image string, medium v1.StorageMedium) {
	var (
		filePath = path.Join(volumePath, "test-file")
		source   = &v1.EmptyDirVolumeSource{Medium: medium}
		pod      = testPodWithVolume(image, volumePath, source)
	)

	pod.Spec.Containers[0].Args = []string{
		fmt.Sprintf("--fs_type=%v", volumePath),
		fmt.Sprintf("--new_file_0644=%v", filePath),
		fmt.Sprintf("--file_perm=%v", filePath),
	}

	msg := fmt.Sprintf("emptydir 0644 on %v", formatMedium(medium))
	out := []string{
		"perms of file \"/test-volume/test-file\": -rw-r--r--",
		"content of file \"/test-volume/test-file\": mount-tester new file",
	}
	if medium == v1.StorageMediumMemory {
		out = append(out, "mount type of \"/test-volume\": tmpfs")
	}
	f.TestContainerOutput(msg, pod, 0, out)
}

func doTest0666(f *framework.Framework, image string, medium v1.StorageMedium) {
	var (
		filePath = path.Join(volumePath, "test-file")
		source   = &v1.EmptyDirVolumeSource{Medium: medium}
		pod      = testPodWithVolume(image, volumePath, source)
	)

	pod.Spec.Containers[0].Args = []string{
		fmt.Sprintf("--fs_type=%v", volumePath),
		fmt.Sprintf("--new_file_0666=%v", filePath),
		fmt.Sprintf("--file_perm=%v", filePath),
	}

	msg := fmt.Sprintf("emptydir 0666 on %v", formatMedium(medium))
	out := []string{
		"perms of file \"/test-volume/test-file\": -rw-rw-rw-",
		"content of file \"/test-volume/test-file\": mount-tester new file",
	}
	if medium == v1.StorageMediumMemory {
		out = append(out, "mount type of \"/test-volume\": tmpfs")
	}
	f.TestContainerOutput(msg, pod, 0, out)
}

func doTest0777(f *framework.Framework, image string, medium v1.StorageMedium) {
	var (
		filePath = path.Join(volumePath, "test-file")
		source   = &v1.EmptyDirVolumeSource{Medium: medium}
		pod      = testPodWithVolume(image, volumePath, source)
	)

	pod.Spec.Containers[0].Args = []string{
		fmt.Sprintf("--fs_type=%v", volumePath),
		fmt.Sprintf("--new_file_0777=%v", filePath),
		fmt.Sprintf("--file_perm=%v", filePath),
	}

	msg := fmt.Sprintf("emptydir 0777 on %v", formatMedium(medium))
	out := []string{
		"perms of file \"/test-volume/test-file\": -rwxrwxrwx",
		"content of file \"/test-volume/test-file\": mount-tester new file",
	}
	if medium == v1.StorageMediumMemory {
		out = append(out, "mount type of \"/test-volume\": tmpfs")
	}
	f.TestContainerOutput(msg, pod, 0, out)
}

func formatMedium(medium v1.StorageMedium) string {
	if medium == v1.StorageMediumMemory {
		return "tmpfs"
	}

	return "node default medium"
}

func testPodWithVolume(image, path string, source *v1.EmptyDirVolumeSource) *v1.Pod {
	podName := "pod-" + string(uuid.NewUUID())
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
						},
					},
				},
			},
			SecurityContext: &v1.PodSecurityContext{
				SELinuxOptions: &v1.SELinuxOptions{
					Level: "s0",
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
			Volumes: []v1.Volume{
				{
					Name: volumeName,
					VolumeSource: v1.VolumeSource{
						EmptyDir: source,
					},
				},
			},
		},
	}
}
