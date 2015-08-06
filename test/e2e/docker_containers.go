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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client"
	"k8s.io/kubernetes/pkg/util"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("Docker Containers", func() {
	var c *client.Client
	var ns string

	BeforeEach(func() {
		var err error
		c, err = loadClient()
		Expect(err).NotTo(HaveOccurred())
		ns_, err := createTestingNS("containers", c)
		ns = ns_.Name
		Expect(err).NotTo(HaveOccurred())
	})

	AfterEach(func() {
		if err := c.Namespaces().Delete(ns); err != nil {
			Failf("Couldn't delete ns %s", err)
		}
	})

	It("should use the image defaults if command and args are blank", func() {
		testContainerOutputInNamespace("use defaults", c, entrypointTestPod(), 0, []string{
			"[/ep default arguments]",
		}, ns)
	})

	It("should be able to override the image's default arguments (docker cmd)", func() {
		pod := entrypointTestPod()
		pod.Spec.Containers[0].Args = []string{"override", "arguments"}

		testContainerOutputInNamespace("override arguments", c, pod, 0, []string{
			"[/ep override arguments]",
		}, ns)
	})

	// Note: when you override the entrypoint, the image's arguments (docker cmd)
	// are ignored.
	It("should be able to override the image's default commmand (docker entrypoint)", func() {
		pod := entrypointTestPod()
		pod.Spec.Containers[0].Command = []string{"/ep-2"}

		testContainerOutputInNamespace("override command", c, pod, 0, []string{
			"[/ep-2]",
		}, ns)
	})

	It("should be able to override the image's default command and arguments", func() {
		pod := entrypointTestPod()
		pod.Spec.Containers[0].Command = []string{"/ep-2"}
		pod.Spec.Containers[0].Args = []string{"override", "arguments"}

		testContainerOutputInNamespace("override all", c, pod, 0, []string{
			"[/ep-2 override arguments]",
		}, ns)
	})
})

const testContainerName = "test-container"

// Return a prototypical entrypoint test pod
func entrypointTestPod() *api.Pod {
	podName := "client-containers-" + string(util.NewUUID())

	return &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: podName,
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:  testContainerName,
					Image: "gcr.io/google_containers/eptest:0.1",
				},
			},
			RestartPolicy: api.RestartPolicyNever,
		},
	}
}
