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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

var _ = framework.KubeDescribe("SimpleMount", func() {
	f := framework.NewDefaultFramework("simple-mount-test")

	// This is a very simple test that exercises the Kubelet's mounter code path.
	// If the mount fails, the pod will not be able to run, and CreateSync will timeout.
	It("should be able to mount an emptydir on a container", func() {
		pod := &v1.Pod{
			TypeMeta: metav1.TypeMeta{
				Kind:       "Pod",
				APIVersion: "v1",
			},
			ObjectMeta: v1.ObjectMeta{
				Name: "simple-mount-pod",
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "simple-mount-container",
						Image: framework.GetPauseImageNameForHostArch(),
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      "simply-mounted-volume",
								MountPath: "/opt/",
							},
						},
					},
				},
				Volumes: []v1.Volume{
					{
						Name: "simply-mounted-volume",
						VolumeSource: v1.VolumeSource{
							EmptyDir: &v1.EmptyDirVolumeSource{
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
