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
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

func runLivenessTest(c *client.Client, podDescr *api.Pod) bool {
	ns := "e2e-test-" + string(util.NewUUID())
	glog.Infof("Creating pod %s in namespace %s", podDescr.Name, ns)
	_, err := c.Pods(ns).Create(podDescr)
	if err != nil {
		glog.Infof("Failed to create pod %s: %v", podDescr.Name, err)
		return false
	}
	// At the end of the test, clean up by removing the pod.
	defer c.Pods(ns).Delete(podDescr.Name)
	// Wait until the pod is not pending. (Here we need to check for something other than
	// 'Pending' other than checking for 'Running', since when failures occur, we go to
	// 'Terminated' which can cause indefinite blocking.)
	err = waitForPodNotPending(c, ns, podDescr.Name, 60*time.Second)
	if err != nil {
		glog.Infof("Failed to start pod %s in namespace %s: %v", podDescr.Name, ns, err)
		return false
	}
	glog.Infof("Started pod %s in namespace %s", podDescr.Name, ns)

	// Check the pod's current state and verify that restartCount is present.
	pod, err := c.Pods(ns).Get(podDescr.Name)
	if err != nil {
		glog.Errorf("Get pod %s in namespace %s failed: %v", podDescr.Name, ns, err)
		return false
	}
	initialRestartCount := pod.Status.Info["liveness"].RestartCount
	glog.Infof("Initial restart count of pod %s is %d", podDescr.Name, initialRestartCount)

	// Wait for at most 48 * 5 = 240s = 4 minutes until restartCount is incremented
	for i := 0; i < 48; i++ {
		// Wait until restartCount is incremented.
		time.Sleep(5 * time.Second)
		pod, err = c.Pods(ns).Get(podDescr.Name)
		if err != nil {
			glog.Errorf("Get pod %s failed: %v", podDescr.Name, err)
			return false
		}
		restartCount := pod.Status.Info["liveness"].RestartCount
		glog.Infof("Restart count of pod %s in namespace %s is now %d", podDescr.Name, ns, restartCount)
		if restartCount > initialRestartCount {
			glog.Infof("Restart count of pod %s in namespace %s increased from %d to %d during the test", podDescr.Name, ns, initialRestartCount, restartCount)
			return true
		}
	}

	glog.Errorf("Did not see the restart count of pod %s in namespace %s increase from %d during the test", podDescr.Name, ns, initialRestartCount)
	return false
}

// TestLivenessHttp tests restarts with a /healthz http liveness probe.
func TestLivenessHttp(c *client.Client) bool {
	return runLivenessTest(c, &api.Pod{
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
}

// TestLivenessExec tests restarts with a docker exec "cat /tmp/health" liveness probe.
func TestLivenessExec(c *client.Client) bool {
	return runLivenessTest(c, &api.Pod{
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
}

var _ = Describe("TestLivenessHttp", func() {
	It("should pass", func() {
		c, err := loadClient()
		Expect(err).NotTo(HaveOccurred())
		Expect(TestLivenessHttp(c)).To(BeTrue())
	})
})

var _ = Describe("TestLivenessExec", func() {
	It("should pass", func() {
		c, err := loadClient()
		Expect(err).NotTo(HaveOccurred())
		Expect(TestLivenessExec(c)).To(BeTrue())
	})
})
