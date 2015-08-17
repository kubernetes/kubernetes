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
	"fmt"
	"strconv"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("MaxPods", func() {
	var c *client.Client
	var nodeCount int
	var totalPodCapacity int64
	var RCName string
	var ns string
	var uuid string

	BeforeEach(func() {
		var err error
		c, err = loadClient()
		expectNoError(err)
		nodes, err := c.Nodes().List(labels.Everything(), fields.Everything())
		expectNoError(err)
		nodeCount = len(nodes.Items)
		Expect(nodeCount).NotTo(BeZero())

		totalPodCapacity = 0
		for _, node := range nodes.Items {
			podCapacity, found := node.Status.Capacity["pods"]
			Expect(found).To(Equal(true))
			totalPodCapacity += podCapacity.Value()
		}

		err = deleteTestingNS(c)
		expectNoError(err)

		nsForTesting, err := createTestingNS("maxp", c)
		ns = nsForTesting.Name
		expectNoError(err)
		uuid = string(util.NewUUID())
	})

	AfterEach(func() {
		rc, err := c.ReplicationControllers(ns).Get(RCName)
		if err == nil && rc.Spec.Replicas != 0 {
			By("Cleaning up the replication controller")
			err := DeleteRC(c, ns, RCName)
			expectNoError(err)
		}

		By(fmt.Sprintf("Destroying namespace for this suite %v", ns))
		if err := c.Namespaces().Delete(ns); err != nil {
			Failf("Couldn't delete ns %s", err)
		}
	})

	// This test verifies that max-pods flag works as advertised. It assumes that cluster add-on pods stay stable
	// and cannot be run in parallel with any other test that touches Nodes or Pods. It is so because to check
	// if max-pods is working we need to fully saturate the cluster and keep it in this state for few seconds.
	It("Validates MaxPods limit number of pods that are allowed to run.", func() {
		pods, err := c.Pods(api.NamespaceAll).List(labels.Everything(), fields.Everything())
		expectNoError(err)
		currentlyRunningPods := len(pods.Items)
		podsNeededForSaturation := int(totalPodCapacity) - currentlyRunningPods

		RCName = "max-pods" + strconv.FormatInt(totalPodCapacity, 10) + "-" + uuid

		config := RCConfig{Client: c,
			Image:        "gcr.io/google_containers/pause:go",
			Name:         RCName,
			Namespace:    ns,
			PollInterval: 10 * time.Second,
			Replicas:     int(podsNeededForSaturation),
		}

		By(fmt.Sprintf("Starting additional %v Pods to fully saturate the cluster and trying to start a new one", podsNeededForSaturation))
		expectNoError(RunRC(config))

		// Log the amount of pods running in the cluster.
		// TODO: remove when test is fixed.
		pods, err = c.Pods(api.NamespaceAll).List(labels.Everything(), fields.Set{
			"status.phase": "Running",
		}.AsSelector())
		Logf("Observed %v running Pods. Need %v to fully saturate the cluster.", len(pods.Items), totalPodCapacity)

		_, err = c.Pods(ns).Create(&api.Pod{
			TypeMeta: api.TypeMeta{
				Kind: "Pod",
			},
			ObjectMeta: api.ObjectMeta{
				Name:   "additional-pod",
				Labels: map[string]string{"name": "additional"},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  "additional-pod",
						Image: "gcr.io/google_containers/pause:go",
					},
				},
			},
		})
		expectNoError(err)
		// Wait a bit to allow scheduler to do its thing
		time.Sleep(10 * time.Second)

		allPods, err := c.Pods(api.NamespaceAll).List(labels.Everything(), fields.Everything())
		expectNoError(err)
		runningPods := 0
		notRunningPods := make([]api.Pod, 0)
		for _, pod := range allPods.Items {
			if pod.Status.Phase == api.PodRunning {
				runningPods += 1
			} else {
				notRunningPods = append(notRunningPods, pod)
			}
		}

		schedEvents, err := c.Events(ns).List(
			labels.Everything(),
			fields.Set{
				"involvedObject.kind":      "Pod",
				"involvedObject.name":      "additional-pod",
				"involvedObject.namespace": ns,
				"source":                   "scheduler",
				"reason":                   "failedScheduling",
			}.AsSelector())
		expectNoError(err)

		Expect(runningPods).To(Equal(int(totalPodCapacity)))
		Expect(len(notRunningPods)).To(Equal(1))
		Expect(schedEvents.Items).ToNot(BeEmpty())
		Expect(notRunningPods[0].Name).To(Equal("additional-pod"))
	})
})
