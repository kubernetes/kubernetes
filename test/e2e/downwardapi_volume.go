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
	"k8s.io/kubernetes/pkg/util"

	. "github.com/onsi/ginkgo"
)

var _ = Describe("Downward API volume", func() {
	f := NewFramework("downward-api")

	It("should provide labels and annotations files", func() {
		podName := "metadata-volume-" + string(util.NewUUID())
		pod := &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name:        podName,
				Labels:      map[string]string{"cluster": "rack10"},
				Annotations: map[string]string{"builder": "john-doe"},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:    "client-container",
						Image:   "gcr.io/google_containers/busybox",
						Command: []string{"sh", "-c", "cat /etc/labels /etc/annotations /etc/podname"},
						VolumeMounts: []api.VolumeMount{
							{
								Name:      "podinfo",
								MountPath: "/etc",
								ReadOnly:  false,
							},
						},
					},
				},
				Volumes: []api.Volume{
					{
						Name: "podinfo",
						VolumeSource: api.VolumeSource{
							DownwardAPI: &api.DownwardAPIVolumeSource{
								Items: []api.DownwardAPIVolumeFile{
									{
										Path: "labels",
										FieldRef: api.ObjectFieldSelector{
											APIVersion: "v1",
											FieldPath:  "metadata.labels",
										},
									},
									{
										Path: "annotations",
										FieldRef: api.ObjectFieldSelector{
											APIVersion: "v1",
											FieldPath:  "metadata.annotations",
										},
									},
									{
										Path: "podname",
										FieldRef: api.ObjectFieldSelector{
											APIVersion: "v1",
											FieldPath:  "metadata.name",
										},
									},
								},
							},
						},
					},
				},
				RestartPolicy: api.RestartPolicyNever,
			},
		}
		testContainerOutputInNamespace("downward API volume plugin", f.Client, pod, 0, []string{
			fmt.Sprintf("cluster=\"rack10\"\n"),
			fmt.Sprintf("builder=\"john-doe\"\n"),
			fmt.Sprintf("%s\n", podName),
		}, f.Namespace.Name)
	})
})

// TODO: add test-webserver example as pointed out in https://github.com/kubernetes/kubernetes/pull/5093#discussion-diff-37606771
