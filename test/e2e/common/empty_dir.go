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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

const (
	testImageRootUid    = "gcr.io/google_containers/mounttest:0.7"
	testImageNonRootUid = "gcr.io/google_containers/mounttest-user:0.3"
)

var _ = framework.KubeDescribe("EmptyDir volumes", func() {

	f := framework.NewDefaultFramework("emptydir")

	Context("when FSGroup is specified [Feature:FSGroup]", func() {
		It("new files should be created with FSGroup ownership when container is root", func() {
			doTestSetgidFSGroup(f, testImageRootUid, api.StorageMediumMemory)
		})

		It("new files should be created with FSGroup ownership when container is non-root", func() {
			doTestSetgidFSGroup(f, testImageNonRootUid, api.StorageMediumMemory)
		})

		It("files with FSGroup ownership should support (root,0644,tmpfs)", func() {
			doTest0644FSGroup(f, testImageRootUid, api.StorageMediumMemory)
		})

		It("volume on default medium should have the correct mode using FSGroup", func() {
			doTestVolumeModeFSGroup(f, testImageRootUid, api.StorageMediumDefault)
		})

		It("volume on tmpfs should have the correct mode using FSGroup", func() {
			doTestVolumeModeFSGroup(f, testImageRootUid, api.StorageMediumMemory)
		})
	})

	It("volume on tmpfs should have the correct mode [Conformance]", func() {
		doTestVolumeMode(f, testImageRootUid, api.StorageMediumMemory)
	})

	It("should support (root,0644,tmpfs) [Conformance]", func() {
		doTest0644(f, testImageRootUid, api.StorageMediumMemory)
	})

	It("should support (root,0666,tmpfs) [Conformance]", func() {
		doTest0666(f, testImageRootUid, api.StorageMediumMemory)
	})

	It("should support (root,0777,tmpfs) [Conformance]", func() {
		doTest0777(f, testImageRootUid, api.StorageMediumMemory)
	})

	It("should support (non-root,0644,tmpfs) [Conformance]", func() {
		doTest0644(f, testImageNonRootUid, api.StorageMediumMemory)
	})

	It("should support (non-root,0666,tmpfs) [Conformance]", func() {
		doTest0666(f, testImageNonRootUid, api.StorageMediumMemory)
	})

	It("should support (non-root,0777,tmpfs) [Conformance]", func() {
		doTest0777(f, testImageNonRootUid, api.StorageMediumMemory)
	})

	It("volume on default medium should have the correct mode [Conformance]", func() {
		doTestVolumeMode(f, testImageRootUid, api.StorageMediumDefault)
	})

	It("should support (root,0644,default) [Conformance]", func() {
		doTest0644(f, testImageRootUid, api.StorageMediumDefault)
	})

	It("should support (root,0666,default) [Conformance]", func() {
		doTest0666(f, testImageRootUid, api.StorageMediumDefault)
	})

	It("should support (root,0777,default) [Conformance]", func() {
		doTest0777(f, testImageRootUid, api.StorageMediumDefault)
	})

	It("should support (non-root,0644,default) [Conformance]", func() {
		doTest0644(f, testImageNonRootUid, api.StorageMediumDefault)
	})

	It("should support (non-root,0666,default) [Conformance]", func() {
		doTest0666(f, testImageNonRootUid, api.StorageMediumDefault)
	})

	It("should support (non-root,0777,default) [Conformance]", func() {
		doTest0777(f, testImageNonRootUid, api.StorageMediumDefault)
	})
})

const (
	containerName = "test-container"
	volumeName    = "test-volume"
)

func doTestSetgidFSGroup(f *framework.Framework, image string, medium api.StorageMedium) {
	var (
		volumePath = "/test-volume"
		filePath   = path.Join(volumePath, "test-file")
		source     = &api.EmptyDirVolumeSource{Medium: medium}
		pod        = testPodWithVolume(testImageRootUid, volumePath, source)
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
	if medium == api.StorageMediumMemory {
		out = append(out, "mount type of \"/test-volume\": tmpfs")
	}
	f.TestContainerOutput(msg, pod, 0, out)
}

func doTestVolumeModeFSGroup(f *framework.Framework, image string, medium api.StorageMedium) {
	var (
		volumePath = "/test-volume"
		source     = &api.EmptyDirVolumeSource{Medium: medium}
		pod        = testPodWithVolume(testImageRootUid, volumePath, source)
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
	if medium == api.StorageMediumMemory {
		out = append(out, "mount type of \"/test-volume\": tmpfs")
	}
	f.TestContainerOutput(msg, pod, 0, out)
}

func doTest0644FSGroup(f *framework.Framework, image string, medium api.StorageMedium) {
	var (
		volumePath = "/test-volume"
		filePath   = path.Join(volumePath, "test-file")
		source     = &api.EmptyDirVolumeSource{Medium: medium}
		pod        = testPodWithVolume(image, volumePath, source)
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
	if medium == api.StorageMediumMemory {
		out = append(out, "mount type of \"/test-volume\": tmpfs")
	}
	f.TestContainerOutput(msg, pod, 0, out)
}

func doTestVolumeMode(f *framework.Framework, image string, medium api.StorageMedium) {
	var (
		volumePath = "/test-volume"
		source     = &api.EmptyDirVolumeSource{Medium: medium}
		pod        = testPodWithVolume(testImageRootUid, volumePath, source)
	)

	pod.Spec.Containers[0].Args = []string{
		fmt.Sprintf("--fs_type=%v", volumePath),
		fmt.Sprintf("--file_perm=%v", volumePath),
	}

	msg := fmt.Sprintf("emptydir volume type on %v", formatMedium(medium))
	out := []string{
		"perms of file \"/test-volume\": -rwxrwxrwx",
	}
	if medium == api.StorageMediumMemory {
		out = append(out, "mount type of \"/test-volume\": tmpfs")
	}
	f.TestContainerOutput(msg, pod, 0, out)
}

func doTest0644(f *framework.Framework, image string, medium api.StorageMedium) {
	var (
		volumePath = "/test-volume"
		filePath   = path.Join(volumePath, "test-file")
		source     = &api.EmptyDirVolumeSource{Medium: medium}
		pod        = testPodWithVolume(image, volumePath, source)
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
	if medium == api.StorageMediumMemory {
		out = append(out, "mount type of \"/test-volume\": tmpfs")
	}
	f.TestContainerOutput(msg, pod, 0, out)
}

func doTest0666(f *framework.Framework, image string, medium api.StorageMedium) {
	var (
		volumePath = "/test-volume"
		filePath   = path.Join(volumePath, "test-file")
		source     = &api.EmptyDirVolumeSource{Medium: medium}
		pod        = testPodWithVolume(image, volumePath, source)
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
	if medium == api.StorageMediumMemory {
		out = append(out, "mount type of \"/test-volume\": tmpfs")
	}
	f.TestContainerOutput(msg, pod, 0, out)
}

func doTest0777(f *framework.Framework, image string, medium api.StorageMedium) {
	var (
		volumePath = "/test-volume"
		filePath   = path.Join(volumePath, "test-file")
		source     = &api.EmptyDirVolumeSource{Medium: medium}
		pod        = testPodWithVolume(image, volumePath, source)
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
	if medium == api.StorageMediumMemory {
		out = append(out, "mount type of \"/test-volume\": tmpfs")
	}
	f.TestContainerOutput(msg, pod, 0, out)
}

func formatMedium(medium api.StorageMedium) string {
	if medium == api.StorageMediumMemory {
		return "tmpfs"
	}

	return "node default medium"
}

func testPodWithVolume(image, path string, source *api.EmptyDirVolumeSource) *api.Pod {
	podName := "pod-" + string(uuid.NewUUID())
	return &api.Pod{
		TypeMeta: unversioned.TypeMeta{
			Kind:       "Pod",
			APIVersion: registered.GroupOrDie(api.GroupName).GroupVersion.String(),
		},
		ObjectMeta: api.ObjectMeta{
			Name: podName,
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:  containerName,
					Image: image,
					VolumeMounts: []api.VolumeMount{
						{
							Name:      volumeName,
							MountPath: path,
						},
					},
				},
			},
			SecurityContext: &api.PodSecurityContext{
				SELinuxOptions: &api.SELinuxOptions{
					Level: "s0",
				},
			},
			RestartPolicy: api.RestartPolicyNever,
			Volumes: []api.Volume{
				{
					Name: volumeName,
					VolumeSource: api.VolumeSource{
						EmptyDir: source,
					},
				},
			},
		},
	}
}
