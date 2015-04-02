/*
Copyright 2015 Google Inc. All rights reserved.

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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	. "github.com/onsi/ginkgo"
)

var _ = Describe("emptyDir", func() {
	var (
		c         *client.Client
		podClient client.PodInterface
	)

	BeforeEach(func() {
		var err error
		c, err = loadClient()
		expectNoError(err)

		podClient = c.Pods(api.NamespaceDefault)
	})

	It("should support tmpfs in emptyDir", func() {
		name := "pod-" + string(util.NewUUID())
		pod := &api.Pod{
			TypeMeta: api.TypeMeta{
				Kind:       "Pod",
				APIVersion: "v1beta1",
			},
			ObjectMeta: api.ObjectMeta{
				Name: name,
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  "mount-test",
						Image: "kubernetes/mounttest:0.1",
						Args:  []string{"--fs_type=/testvol"},
						VolumeMounts: []api.VolumeMount{
							{
								Name:      "testvol",
								MountPath: "/testvol",
							},
						},
					},
				},
				Volumes: []api.Volume{
					{
						Name: "testvol",
						VolumeSource: api.VolumeSource{
							EmptyDir: &api.EmptyDirVolumeSource{
								Medium: api.StorageTypeMemory,
							},
						},
					},
				},
			},
		}

		testContainerOutput("tmpfs mount for emptydir", c, pod, []string{
			"mount type: tmpfs",
		})
	})
})
