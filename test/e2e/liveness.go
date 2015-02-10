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

// Tests for liveness probes, both with http and with docker exec.
// These tests use the descriptions in examples/liveness to create test pods.

package e2e

import (
	"fmt"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

func runLivenessTest(c *client.Client, podDescr *api.Pod) {
	defer GinkgoRecover()
	ns := "e2e-test-" + string(util.NewUUID())

	By(fmt.Sprintf("Creating pod %s in namespace %s", podDescr.Name, ns))
	_, err := c.Pods(ns).Create(podDescr)
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("creating pod %s", podDescr.Name))

	// At the end of the test, clean up by removing the pod.
	defer func() {
		By("deleting the pod")
		defer GinkgoRecover()
		c.Pods(ns).Delete(podDescr.Name)
	}()

	// Wait until the pod is not pending. (Here we need to check for something other than
	// 'Pending' other than checking for 'Running', since when failures occur, we go to
	// 'Terminated' which can cause indefinite blocking.)
	By("waiting for the pod to be something other than pending")
	err = waitForPodNotPending(c, ns, podDescr.Name, 60*time.Second)
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("starting pod %s in namespace %s", podDescr.Name, ns))
	fmt.Printf("Started pod %s in namespace %s\n", podDescr.Name, ns)

	// Check the pod's current state and verify that restartCount is present.
	By("checking the pod's current state and verifying that restartCount is present")
	pod, err := c.Pods(ns).Get(podDescr.Name)
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("getting pod %s in namespace %s", podDescr.Name, ns))
	initialRestartCount := pod.Status.Info["liveness"].RestartCount
	fmt.Printf("Initial restart count of pod %s is %d\n", podDescr.Name, initialRestartCount)

	// Wait for at most 48 * 5 = 240s = 4 minutes until restartCount is incremented
	pass := false
	for i := 0; i < 48; i++ {
		// Wait until restartCount is incremented.
		time.Sleep(5 * time.Second)
		pod, err = c.Pods(ns).Get(podDescr.Name)
		Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("getting pod %s", podDescr.Name))
		restartCount := pod.Status.Info["liveness"].RestartCount
		fmt.Printf("Restart count of pod %s in namespace %s is now %d\n", podDescr.Name, ns, restartCount)
		if restartCount > initialRestartCount {
			fmt.Printf("Restart count of pod %s in namespace %s increased from %d to %d during the test\n", podDescr.Name, ns, initialRestartCount, restartCount)
			pass = true
			break
		}
	}

	if !pass {
		Fail(fmt.Sprintf("Did not see the restart count of pod %s in namespace %s increase from %d during the test", podDescr.Name, ns, initialRestartCount))
	}
}

var _ = Describe("TestLivenessHttp", func() {
	var c *client.Client

	BeforeEach(func() {
		var err error
		c, err = loadClient()
		Expect(err).NotTo(HaveOccurred())
	})

	It("should restart pods with a /healthz http liveness probe", func() {
		runLivenessTest(c, &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name:   "liveness-http",
				Labels: map[string]string{"test": "liveness"},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:    "liveness",
						Image:   "kubernetes/liveness",
						Command: []string{"/server"},
						LivenessProbe: &api.Probe{
							Handler: api.Handler{
								HTTPGet: &api.HTTPGetAction{
									Path: "/healthz",
									Port: util.NewIntOrStringFromInt(8080),
								},
							},
							InitialDelaySeconds: 15,
						},
					},
				},
			},
		})
	})
})

var _ = Describe("TestLivenessExec", func() {
	var c *client.Client

	BeforeEach(func() {
		var err error
		c, err = loadClient()
		Expect(err).NotTo(HaveOccurred())
	})

	It("should restart pods with a docker exec \"cat /tmp/health\"", func() {
		runLivenessTest(c, &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name:   "liveness-exec",
				Labels: map[string]string{"test": "liveness"},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:    "liveness",
						Image:   "busybox",
						Command: []string{"/bin/sh", "-c", "echo ok >/tmp/health; sleep 10; echo fail >/tmp/health; sleep 600"},
						LivenessProbe: &api.Probe{
							Handler: api.Handler{
								Exec: &api.ExecAction{
									Command: []string{"cat", "/tmp/health"},
								},
							},
							InitialDelaySeconds: 15,
						},
					},
				},
			},
		})
	})
})
