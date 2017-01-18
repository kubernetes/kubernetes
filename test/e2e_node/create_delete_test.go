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

var _ = framework.KubeDescribe("CreateDelete", func() {
	f := framework.NewDefaultFramework("create-delete-test")
	testPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod-with-volume",
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers: []v1.Container{
				{
					Image: "gcr.io/google_containers/busybox:1.24",
					Name:  "volume-pod",
					Command: []string{
						"sh",
						"-c",
						"sleep 10000",
					},
					VolumeMounts: []v1.VolumeMount{
						{MountPath: "/test-empty-dir-mnt", Name: "test-empty-dir"},
					},
				},
			},
			Volumes: []v1.Volume{
				{Name: "test-empty-dir", VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}}},
			},
		},
	}

	Context("When we make a pod with a volume", func() {
		It("Should successfully create and delete the pod", func() {
			f.PodClient().CreateSync(testPod)
			f.PodClient().DeleteSync(testPod.Name, &v1.DeleteOptions{}, podDisappearTimeout)
		})
	})
})
