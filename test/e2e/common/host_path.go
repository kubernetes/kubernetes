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
	"os"
	"path"

	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

//TODO : Consolidate this code with the code for emptyDir.
//This will require some smart.
var _ = framework.KubeDescribe("HostPath", func() {
	f := framework.NewDefaultFramework("hostpath")

	BeforeEach(func() {
		//cleanup before running the test.
		_ = os.Remove("/tmp/test-file")
	})

	It("should give a volume the correct mode [Conformance]", func() {
		volumePath := "/test-volume"
		source := &v1.HostPathVolumeSource{
			Path: "/tmp",
		}
		pod := testPodWithHostVol(volumePath, source)

		pod.Spec.Containers[0].Args = []string{
			fmt.Sprintf("--fs_type=%v", volumePath),
			fmt.Sprintf("--file_mode=%v", volumePath),
		}
		f.TestContainerOutput("hostPath mode", pod, 0, []string{
			"mode of file \"/test-volume\": dtrwxrwxrwx", // we expect the sticky bit (mode flag t) to be set for the dir
		})
	})

	// This test requires mounting a folder into a container with write privileges.
	It("should support r/w", func() {
		volumePath := "/test-volume"
		filePath := path.Join(volumePath, "test-file")
		retryDuration := 180
		source := &v1.HostPathVolumeSource{
			Path: "/tmp",
		}
		pod := testPodWithHostVol(volumePath, source)

		pod.Spec.Containers[0].Args = []string{
			fmt.Sprintf("--new_file_0644=%v", filePath),
			fmt.Sprintf("--file_mode=%v", filePath),
		}

		pod.Spec.Containers[1].Args = []string{
			fmt.Sprintf("--file_content_in_loop=%v", filePath),
			fmt.Sprintf("--retry_time=%d", retryDuration),
		}
		//Read the content of the file with the second container to
		//verify volumes  being shared properly among containers within the pod.
		f.TestContainerOutput("hostPath r/w", pod, 1, []string{
			"content of file \"/test-volume/test-file\": mount-tester new file",
		})
	})

	It("should support subPath", func() {
		volumePath := "/test-volume"
		subPath := "sub-path"
		fileName := "test-file"
		retryDuration := 180

		filePathInWriter := path.Join(volumePath, fileName)
		filePathInReader := path.Join(volumePath, subPath, fileName)

		source := &v1.HostPathVolumeSource{
			Path: "/tmp",
		}
		pod := testPodWithHostVol(volumePath, source)
		// Write the file in the subPath from container 0
		container := &pod.Spec.Containers[0]
		container.VolumeMounts[0].SubPath = subPath
		container.Args = []string{
			fmt.Sprintf("--new_file_0644=%v", filePathInWriter),
			fmt.Sprintf("--file_mode=%v", filePathInWriter),
		}
		// Read it from outside the subPath from container 1
		pod.Spec.Containers[1].Args = []string{
			fmt.Sprintf("--file_content_in_loop=%v", filePathInReader),
			fmt.Sprintf("--retry_time=%d", retryDuration),
		}

		f.TestContainerOutput("hostPath subPath", pod, 1, []string{
			"content of file \"" + filePathInReader + "\": mount-tester new file",
		})
	})
})

//These constants are borrowed from the other test.
//const volumeName = "test-volume"
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

//TODO: To merge this with the emptyDir tests, we can make source a lambda.
func testPodWithHostVol(path string, source *v1.HostPathVolumeSource) *v1.Pod {
	podName := "pod-host-path-test"

	return &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: registered.GroupOrDie(v1.GroupName).GroupVersion.String(),
		},
		ObjectMeta: v1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  containerName1,
					Image: "gcr.io/google_containers/mounttest:0.7",
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      volumeName,
							MountPath: path,
						},
					},
				},
				{
					Name:  containerName2,
					Image: "gcr.io/google_containers/mounttest:0.7",
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      volumeName,
							MountPath: path,
						},
					},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
			Volumes:       mount(source),
		},
	}
}
