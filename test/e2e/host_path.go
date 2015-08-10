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
	"k8s.io/kubernetes/pkg/client"
	"k8s.io/kubernetes/pkg/util"
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
		if err := c.Namespaces().Delete(namespace.Name); err != nil {
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
		source := &api.HostPathVolumeSource{
			Path: "/tmp",
		}
		pod := testPodWithHostVol(volumePath, source)

		pod.Spec.Containers[0].Args = []string{
			fmt.Sprintf("--fs_type=%v", volumePath),
			fmt.Sprintf("--rw_new_file=%v", filePath),
			fmt.Sprintf("--file_mode=%v", filePath),
		}
		testContainerOutputInNamespace("hostPath r/w", c, pod, 0, []string{
			"mode of file \"/test-volume/test-file\": -rw-r--r--",
			"content of file \"/test-volume/test-file\": mount-tester new file",
		}, namespace.Name,
		)
	})
})

//These constants are borrowed from the other test.
//const containerName = "test-container"
//const volumeName = "test-volume"

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
					Image: "gcr.io/google_containers/mounttest:0.2",
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
