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

package storage

import (
	"context"
	"fmt"
	"os"
	"path"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epodoutput "k8s.io/kubernetes/test/e2e/framework/pod/output"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
)

// TODO : Consolidate this code with the code for emptyDir.
// This will require some smart.
var _ = SIGDescribe("HostPath", func() {
	f := framework.NewDefaultFramework("hostpath")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.BeforeEach(func() {
		// TODO permission denied cleanup failures
		//cleanup before running the test.
		_ = os.Remove("/tmp/test-file")
	})

	/*
	   Host path, volume mode default
	   Create a Pod with host volume mounted. The volume mounted MUST be a directory with permissions mode -rwxrwxrwx and that is has the sticky bit (mode flag t) set.
	   This test is marked LinuxOnly since Windows does not support setting the sticky bit (mode flag t).
	*/
	f.It("should give a volume the correct mode [LinuxOnly]", f.WithNodeConformance(), func(ctx context.Context) {
		source := &v1.HostPathVolumeSource{
			Path: "/tmp",
		}
		pod := testPodWithHostVol(volumePath, source, false)

		pod.Spec.Containers[0].Args = []string{
			"mounttest",
			fmt.Sprintf("--fs_type=%v", volumePath),
			fmt.Sprintf("--file_mode=%v", volumePath),
		}
		e2epodoutput.TestContainerOutputRegexp(ctx, f, "hostPath mode", pod, 0, []string{
			"mode of file \"/test-volume\": dg?trwxrwx", // we expect the sticky bit (mode flag t) to be set for the dir
		})
	})

	// This test requires mounting a folder into a container with write privileges.
	f.It("should support r/w", f.WithNodeConformance(), func(ctx context.Context) {
		filePath := path.Join(volumePath, "test-file")
		retryDuration := 180
		source := &v1.HostPathVolumeSource{
			Path: "/tmp",
		}
		// we can't spawn privileged containers on Windows, nor do we need to.
		privileged := !framework.NodeOSDistroIs("windows")
		pod := testPodWithHostVol(volumePath, source, privileged)

		pod.Spec.Containers[0].Args = []string{
			"mounttest",
			fmt.Sprintf("--new_file_0644=%v", filePath),
			fmt.Sprintf("--file_mode=%v", filePath),
		}

		pod.Spec.Containers[1].Args = []string{
			"mounttest",
			fmt.Sprintf("--file_content_in_loop=%v", filePath),
			fmt.Sprintf("--retry_time=%d", retryDuration),
		}
		//Read the content of the file with the second container to
		//verify volumes  being shared properly among containers within the pod.
		e2epodoutput.TestContainerOutput(ctx, f, "hostPath r/w", pod, 1, []string{
			"content of file \"/test-volume/test-file\": mount-tester new file",
		})
	})

	f.It("should support subPath", f.WithNodeConformance(), func(ctx context.Context) {
		subPath := "sub-path"
		fileName := "test-file"
		retryDuration := 180

		filePathInWriter := path.Join(volumePath, fileName)
		filePathInReader := path.Join(volumePath, subPath, fileName)

		source := &v1.HostPathVolumeSource{
			Path: "/tmp",
		}

		// we can't spawn privileged containers on Windows, nor do we need to.
		privileged := !framework.NodeOSDistroIs("windows")
		pod := testPodWithHostVol(volumePath, source, privileged)

		// Write the file in the subPath from container 0
		container := &pod.Spec.Containers[0]
		container.VolumeMounts[0].SubPath = subPath
		container.Args = []string{
			"mounttest",
			fmt.Sprintf("--new_file_0644=%v", filePathInWriter),
			fmt.Sprintf("--file_mode=%v", filePathInWriter),
		}

		// Read it from outside the subPath from container 1
		pod.Spec.Containers[1].Args = []string{
			"mounttest",
			fmt.Sprintf("--file_content_in_loop=%v", filePathInReader),
			fmt.Sprintf("--retry_time=%d", retryDuration),
		}

		e2epodoutput.TestContainerOutput(ctx, f, "hostPath subPath", pod, 1, []string{
			"content of file \"" + filePathInReader + "\": mount-tester new file",
		})
	})
})

// These constants are borrowed from the other test.
// const volumeName = "test-volume"
const containerName1 = "test-container-1"
const containerName2 = "test-container-2"

func mount(source *v1.HostPathVolumeSource) []v1.Volume {
	return []v1.Volume{
		{
			Name: volumeName,
			VolumeSource: v1.VolumeSource{
				HostPath: source,
			},
		},
	}
}

// TODO: To merge this with the emptyDir tests, we can make source a lambda.
func testPodWithHostVol(path string, source *v1.HostPathVolumeSource, privileged bool) *v1.Pod {
	podName := "pod-host-path-test"

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
					Name:  containerName1,
					Image: imageutils.GetE2EImage(imageutils.Agnhost),
					Args:  []string{"mounttest"},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      volumeName,
							MountPath: path,
						},
					},
					SecurityContext: &v1.SecurityContext{
						Privileged: &privileged,
					},
				},
				{
					Name:  containerName2,
					Image: imageutils.GetE2EImage(imageutils.Agnhost),
					Args:  []string{"mounttest"},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      volumeName,
							MountPath: path,
						},
					},
					SecurityContext: &v1.SecurityContext{
						Privileged: &privileged,
					},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
			Volumes:       mount(source),
		},
	}
}
