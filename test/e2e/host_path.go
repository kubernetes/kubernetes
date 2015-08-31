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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/latest"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"os"
	"path"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

//TODO : Consolidate this code with the code for emptyDir.
//This will require some smart.
var _ = Describe("hostPath", func() {
	var (
		c         *client.Client
		namespace *api.Namespace
	)

	BeforeEach(func() {
		var err error
		c, err = loadClient()
		Expect(err).NotTo(HaveOccurred())

		By("Building a namespace api object")
		namespace, err = createTestingNS("hostpath", c)
		Expect(err).NotTo(HaveOccurred())

		//cleanup before running the test.
		_ = os.Remove("/tmp/test-file")
	})

	AfterEach(func() {
		By(fmt.Sprintf("Destroying namespace for this suite %v", namespace.Name))
		if err := deleteNS(c, namespace.Name); err != nil {
			Failf("Couldn't delete ns %s", err)
		}
	})

	It("should give a volume the correct mode", func() {
		volumePath := "/test-volume"
		source := &api.HostPathVolumeSource{
			Path: "/tmp",
		}
		pod := testPodWithHostVol(volumePath, source)

		pod.Spec.Containers[0].Args = []string{
			fmt.Sprintf("--fs_type=%v", volumePath),
			fmt.Sprintf("--file_mode=%v", volumePath),
		}
		testContainerOutputInNamespace("hostPath mode", c, pod, 0, []string{
			"mode of file \"/test-volume\": dtrwxrwxrwx", // we expect the sticky bit (mode flag t) to be set for the dir
		},
			namespace.Name)
	})

	It("should support r/w", func() {
		volumePath := "/test-volume"
		filePath := path.Join(volumePath, "test-file")
		retryDuration := 180
		source := &api.HostPathVolumeSource{
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
		//verify volumes  being shared properly among continers within the pod.
		testContainerOutputInNamespace("hostPath r/w", c, pod, 1, []string{
			"content of file \"/test-volume/test-file\": mount-tester new file",
		}, namespace.Name,
		)
	})
})

//These constants are borrowed from the other test.
//const volumeName = "test-volume"
const containerName1 = "test-container-1"
const containerName2 = "test-container-2"

func mount(source *api.HostPathVolumeSource) []api.Volume {
	return []api.Volume{
		{
			Name: volumeName,
			VolumeSource: api.VolumeSource{
				HostPath: source,
			},
		},
	}
}

//TODO: To merge this with the emptyDir tests, we can make source a lambda.
func testPodWithHostVol(path string, source *api.HostPathVolumeSource) *api.Pod {
	podName := "pod-host-path-test"

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
					Image: "gcr.io/google_containers/mounttest:0.4",
					VolumeMounts: []api.VolumeMount{
						{
							Name:      volumeName,
							MountPath: path,
						},
					},
				},
				{
					Name:  containerName2,
					Image: "gcr.io/google_containers/mounttest:0.4",
					VolumeMounts: []api.VolumeMount{
						{
							Name:      volumeName,
							MountPath: path,
						},
					},
				},
			},
			RestartPolicy: api.RestartPolicyNever,
			Volumes:       mount(source),
		},
	}
}
