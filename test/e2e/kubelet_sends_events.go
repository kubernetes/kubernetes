/*
Copyright 2014 Google Inc. All rights reserved.

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
	"strconv"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/golang/glog"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// TestKubeletSendsEvent checks that kubelets and scheduler send events about pods scheduling and running.
func TestKubeletSendsEvent(c *client.Client) bool {
	provider := testContext.provider
	if len(provider) > 0 && provider != "gce" && provider != "gke" {
		glog.Infof("skipping TestKubeletSendsEvent on cloud provider %s", provider)
		return true
	}

	podClient := c.Pods(api.NamespaceDefault)

	pod := loadPodOrDie(assetPath("test", "e2e", "pod.json"))
	value := strconv.Itoa(time.Now().Nanosecond())
	pod.Labels["time"] = value

	_, err := podClient.Create(pod)
	if err != nil {
		glog.Errorf("Failed to create pod: %v", err)
		return false
	}
	defer podClient.Delete(pod.Name)
	waitForPodRunning(c, pod.Name)
	pods, err := podClient.List(labels.SelectorFromSet(labels.Set(map[string]string{"time": value})))
	if len(pods.Items) != 1 {
		glog.Errorf("Failed to find the correct pod")
		return false
	}

	podWithUid, err := podClient.Get(pod.Name)
	if err != nil {
		glog.Errorf("Failed to get pod: %v", err)
		return false
	}

	// Check for scheduler event about the pod.
	glog.Infof("%+v", podWithUid)
	events, err := c.Events(api.NamespaceDefault).List(
		labels.Everything(),
		labels.Set{
			"involvedObject.kind":      "Pod",
			"involvedObject.uid":       string(podWithUid.UID),
			"involvedObject.namespace": api.NamespaceDefault,
			"source":                   "scheduler",
		}.AsSelector(),
	)
	if err != nil {
		glog.Error("Error while listing events:", err)
		return false
	}
	if len(events.Items) == 0 {
		glog.Error("Didn't see any scheduler events even though pod was running.")
		return false
	}
	glog.Info("Saw scheduler event for our pod.")

	// Check for kubelet event about the pod.
	events, err = c.Events(api.NamespaceDefault).List(
		labels.Everything(),
		labels.Set{
			"involvedObject.uid":       string(podWithUid.UID),
			"involvedObject.kind":      "BoundPod",
			"involvedObject.namespace": api.NamespaceDefault,
			"source":                   "kubelet",
		}.AsSelector(),
	)
	if err != nil {
		glog.Error("Error while listing events:", err)
		return false
	}
	if len(events.Items) == 0 {
		glog.Error("Didn't see any kubelet events even though pod was running.")
		return false
	}
	glog.Info("Saw kubelet event for our pod.")
	return true
}

var _ = Describe("TestKubeletSendsEvent", func() {
	It("should pass", func() {
		// TODO: Instead of OrDie, client should Fail the test if there's a problem.
		// In general tests should Fail() instead of glog.Fatalf().
		Expect(TestKubeletSendsEvent(loadClientOrDie())).To(BeTrue())
	})
})
