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

package e2e_node

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

var _ = framework.KubeDescribe("SimpleMount", func() {
	f := framework.NewDefaultFramework("simple-mount-test")

	// This is a very simple test that exercises the Kubelet's mounter code path.
	// If the mount fails, the pod will not be able to run, and CreateSync will timeout.
	It("should be able to mount an emptydir on a container", func() {
		pod := &api.Pod{
			TypeMeta: unversioned.TypeMeta{
				Kind:       "Pod",
				APIVersion: "v1",
			},
			ObjectMeta: api.ObjectMeta{
				Name: "simple-mount-pod",
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  "simple-mount-container",
						Image: framework.GetPauseImageNameForHostArch(),
						VolumeMounts: []api.VolumeMount{
							{
								Name:      "simply-mounted-volume",
								MountPath: "/opt/",
							},
						},
					},
				},
				Volumes: []api.Volume{
					{
						Name: "simply-mounted-volume",
						VolumeSource: api.VolumeSource{
							EmptyDir: &api.EmptyDirVolumeSource{
								Medium: "Memory",
							},
						},
					},
				},
			},
		}
		podClient := f.PodClient()
		pod = podClient.CreateSync(pod)

	})
})
