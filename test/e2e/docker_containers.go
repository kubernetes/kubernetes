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
	"fmt"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("Docker Containers", func() {
	var c *client.Client

	BeforeEach(func() {
		var err error
		c, err = loadClient()
		Expect(err).NotTo(HaveOccurred())
	})

	It("should use the image defaults if command and args are blank", func() {
		runEntrypointTest("use defaults", c, entrypointTestPod(), []string{
			"[/ep default arguments]",
		})
	})

	It("should be able to override the image's default arguments (docker cmd)", func() {
		pod := entrypointTestPod()
		pod.Spec.Containers[0].Args = []string{"override", "arguments"}

		runEntrypointTest("override arguments", c, pod, []string{
			"[/ep override arguments]",
		})
	})

	// Note: when you override the entrypoint, the image's arguments (docker cmd)
	// are ignored.
	It("should be able to override the image's default commmand (docker entrypoint)", func() {
		pod := entrypointTestPod()
		pod.Spec.Containers[0].Command = []string{"/ep-2"}

		runEntrypointTest("override command", c, pod, []string{
			"[/ep-2]",
		})
	})

	It("should be able to override the image's default command and arguments", func() {
		pod := entrypointTestPod()
		pod.Spec.Containers[0].Command = []string{"/ep-2"}
		pod.Spec.Containers[0].Args = []string{"override", "arguments"}

		runEntrypointTest("override all", c, pod, []string{
			"[/ep-2 override arguments]",
		})
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
					Image: "kubernetes/eptest:0.1",
				},
			},
			RestartPolicy: api.RestartPolicyNever,
		},
	}
}

// pod must have a container named 'test-container'
func runEntrypointTest(scenarioName string, c *client.Client, pod *api.Pod, expectedOutput []string) {
	ns := api.NamespaceDefault
	By(fmt.Sprintf("Creating a pod to test %v", scenarioName))

	defer c.Pods(ns).Delete(pod.Name)
	if _, err := c.Pods(ns).Create(pod); err != nil {
		Failf("Failed to create pod: %v", err)
	}
	// Wait for client pod to complete.
	expectNoError(waitForPodSuccess(c, pod.Name, testContainerName))

	// Grab its logs.  Get host first.
	podStatus, err := c.Pods(ns).Get(pod.Name)
	if err != nil {
		Failf("Failed to get pod to know host: %v", err)
	}
	By(fmt.Sprintf("Trying to get logs from host %s pod %s container %s: %v",
		podStatus.Status.Host, podStatus.Name, podStatus.Spec.Containers[0].Name, err))
	var logs []byte
	start := time.Now()

	// Sometimes the actual containers take a second to get started, try to get logs for 60s
	for time.Now().Sub(start) < (60 * time.Second) {
		logs, err = c.Get().
			Prefix("proxy").
			Resource("minions").
			Name(podStatus.Status.Host).
			Suffix("containerLogs", ns, podStatus.Name, podStatus.Spec.Containers[0].Name).
			Do().
			Raw()
		fmt.Sprintf("pod logs:%v\n", string(logs))
		By(fmt.Sprintf("pod logs:%v\n", string(logs)))
		if strings.Contains(string(logs), "Internal Error") {
			By(fmt.Sprintf("Failed to get logs from host %s pod %s container %s: %v",
				podStatus.Status.Host, podStatus.Name, podStatus.Spec.Containers[0].Name, string(logs)))
			time.Sleep(5 * time.Second)
			continue
		}
		break
	}

	for _, m := range expectedOutput {
		Expect(string(logs)).To(ContainSubstring(m), "%q in container output", m)
	}
}
