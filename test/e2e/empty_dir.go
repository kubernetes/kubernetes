/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package e2e

import (
	"fmt"
	"path"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	. "github.com/onsi/ginkgo"
)

var _ = Describe("emptyDir", func() {
	var (
		c *client.Client
	)

	BeforeEach(func() {
		var err error
		c, err = loadClient()
		expectNoError(err)
	})

	It("volume on tmpfs should have the correct mode", func() {
		volumePath := "/test-volume"
		source := &api.EmptyDirVolumeSource{
			Medium: api.StorageTypeMemory,
		}
		pod := testPodWithVolume(volumePath, source)

		pod.Spec.Containers[0].Args = []string{
			fmt.Sprintf("--fs_type=%v", volumePath),
			fmt.Sprintf("--file_mode=%v", volumePath),
		}
		testContainerOutput("emptydir r/w on tmpfs", c, pod, []string{
			"mount type of \"/test-volume\": tmpfs",
			"mode of file \"/test-volume\": dtrwxrwxrwx", // we expect the sticky bit (mode flag t) to be set for the dir
		})
	})

	It("should support r/w on tmpfs", func() {
		volumePath := "/test-volume"
		filePath := path.Join(volumePath, "test-file")
		source := &api.EmptyDirVolumeSource{
			Medium: api.StorageTypeMemory,
		}
		pod := testPodWithVolume(volumePath, source)

		pod.Spec.Containers[0].Args = []string{
			fmt.Sprintf("--fs_type=%v", volumePath),
			fmt.Sprintf("--rw_new_file=%v", filePath),
			fmt.Sprintf("--file_mode=%v", filePath),
		}
		testContainerOutput("emptydir r/w on tmpfs", c, pod, []string{
			"mount type of \"/test-volume\": tmpfs",
			"mode of file \"/test-volume/test-file\": -rw-r--r--",
			"content of file \"/test-volume/test-file\": mount-tester new file",
		})
	})
})

const containerName = "test-container"
const volumeName = "test-volume"

func testPodWithVolume(path string, source *api.EmptyDirVolumeSource) *api.Pod {
	podName := "pod-" + string(util.NewUUID())

	return &api.Pod{
		TypeMeta: api.TypeMeta{
			Kind:       "Pod",
			APIVersion: latest.Version,
		},
		ObjectMeta: api.ObjectMeta{
			Name: podName,
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:  containerName,
					Image: "kubernetes/mounttest:0.1",
					VolumeMounts: []api.VolumeMount{
						{
							Name:      volumeName,
							MountPath: path,
						},
					},
				},
			},
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
