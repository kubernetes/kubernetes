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
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

var _ = framework.KubeDescribe("Docker Containers", func() {
	f := framework.NewDefaultFramework("containers")

	It("should use the image defaults if command and args are blank [Conformance]", func() {
		f.TestContainerOutput("use defaults", entrypointTestPod(), 0, []string{
			"[/ep default arguments]",
		})
	})

	It("should be able to override the image's default arguments (docker cmd) [Conformance]", func() {
		pod := entrypointTestPod()
		pod.Spec.Containers[0].Args = []string{"override", "arguments"}

		f.TestContainerOutput("override arguments", pod, 0, []string{
			"[/ep override arguments]",
		})
	})

	// Note: when you override the entrypoint, the image's arguments (docker cmd)
	// are ignored.
	It("should be able to override the image's default commmand (docker entrypoint) [Conformance]", func() {
		pod := entrypointTestPod()
		pod.Spec.Containers[0].Command = []string{"/ep-2"}

		f.TestContainerOutput("override command", pod, 0, []string{
			"[/ep-2]",
		})
	})

	It("should be able to override the image's default command and arguments [Conformance]", func() {
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

	return &v1.Pod{
		ObjectMeta: v1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  testContainerName,
					Image: "gcr.io/google_containers/eptest:0.1",
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
		},
	}
}
