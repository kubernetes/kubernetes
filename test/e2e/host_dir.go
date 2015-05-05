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

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("HostDir", func() {
	var (
		c         *client.Client = nil
		namespace *api.Namespace
	)

	BeforeEach(func() {
		var err error
		c, err = loadClient()
		expectNoError(err)

		By("Building a namespace api object")
		namespace, err = createTestingNS("hostdir-test", c)
		Expect(err).NotTo(HaveOccurred())

	})

	AfterEach(func() {
		By(fmt.Sprintf("Destroying namespace for this suite %v", namespace.Name))
		if err := c.Namespaces().Delete(namespace.Name); err != nil {
			Failf("Couldn't delete ns %s", err)
		}
	})

	It("should support r/w", func() {
		volumePath := "/home"

		source := &api.HostPathVolumeSource{
			Path: "/home",
		}
		containerfilePath := path.Join(source.Path, "hostdir-test-file")

		pod := testPodWithHostVolume(volumePath, source)
		pod.Spec.Containers[0].Args = []string{
			fmt.Sprintf("--rw_new_file=%v", containerfilePath),
		}
		pod.Spec.Containers[1].Args = []string{
			fmt.Sprintf("--file_content_in_loop=%v", containerfilePath),
		}
		testContainerOutputInNamespace("hostdir support r/w", c, pod, 1, []string{
			"content of file \"/home/hostdir-test-file\": mount-tester new file",
		}, namespace.Name)
	})

})

const containerName1 = "test-container-1"
const containerName2 = "test-container-2"
const hostdirvolumeName = "test-volume"

func testPodWithHostVolume(path string, source *api.HostPathVolumeSource) *api.Pod {
	podName := "pod-host-dir-test"

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
					Name:  containerName1,
					Image: "kubernetes/mounttest:0.1",
					VolumeMounts: []api.VolumeMount{
						{
							Name:      hostdirvolumeName,
							MountPath: path,
						},
					},
				},
				{
					Name:  containerName2,
					Image: "kubernetes/mounttest:0.1",
					VolumeMounts: []api.VolumeMount{
						{
							Name:      hostdirvolumeName,
							MountPath: path,
						},
					},
				},
			},
			Volumes: []api.Volume{
				{
					Name: hostdirvolumeName,
					VolumeSource: api.VolumeSource{
						HostPath: source,
					},
				},
			},
		},
	}
}
