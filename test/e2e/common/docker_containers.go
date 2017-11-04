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
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

var _ = framework.KubeDescribe("Docker Containers", func() {
	f := framework.NewDefaultFramework("containers")

	framework.ConformanceIt("should use the image defaults if command and args are blank ", func() {
		f.TestContainerOutput("use defaults", entrypointTestPod(), 0, []string{
			"[/ep default arguments]",
		})
	})

	framework.ConformanceIt("should be able to override the image's default arguments (docker cmd) ", func() {
		pod := entrypointTestPod()
		pod.Spec.Containers[0].Args = []string{"override", "arguments"}

		f.TestContainerOutput("override arguments", pod, 0, []string{
			"[/ep override arguments]",
		})
	})

	// Note: when you override the entrypoint, the image's arguments (docker cmd)
	// are ignored.
	framework.ConformanceIt("should be able to override the image's default commmand (docker entrypoint) ", func() {
		pod := entrypointTestPod()
		pod.Spec.Containers[0].Command = []string{"/ep-2"}

		f.TestContainerOutput("override command", pod, 0, []string{
			"[/ep-2]",
		})
	})

	framework.ConformanceIt("should be able to override the image's default command and arguments ", func() {
		pod := entrypointTestPod()
		pod.Spec.Containers[0].Command = []string{"/ep-2"}
		pod.Spec.Containers[0].Args = []string{"override", "arguments"}

		f.TestContainerOutput("override all", pod, 0, []string{
			"[/ep-2 override arguments]",
		})
	})
})

const testContainerName = "test-container"

// Return a prototypical entrypoint test pod
func entrypointTestPod() *v1.Pod {
	podName := "client-containers-" + string(uuid.NewUUID())

	one := int64(1)
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  testContainerName,
					Image: imageutils.GetE2EImage(imageutils.EntrypointTester),
				},
			},
			RestartPolicy:                 v1.RestartPolicyNever,
			TerminationGracePeriodSeconds: &one,
		},
	}
}
