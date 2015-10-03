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
	"time"

	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/sets"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	// Interval to poll /stats/container on a node
	containerStatsPollingPeriod = 10 * time.Second
	// The monitoring time for one test.
	monitoringTime = 30 * time.Minute
	// The periodic reporting period.
	reportingPeriod = 5 * time.Minute
)

func logPodsOnNodes(c *client.Client, nodeNames []string) {
	for _, n := range nodeNames {
		podList, err := GetKubeletPods(c, n)
		if err != nil {
			Logf("Unable to retrieve kubelet pods for node %v", n)
			continue
		}
		Logf("%d pods are running on node %v", len(podList.Items), n)
	}
}

func runResourceTrackingTest(framework *Framework, podsPerNode int, nodeNames sets.String, resourceMonitor *resourceMonitor) {
	numNodes := nodeNames.Len()
	totalPods := podsPerNode * numNodes
	By(fmt.Sprintf("Creating a RC of %d pods and wait until all pods of this RC are running", totalPods))
	rcName := fmt.Sprintf("resource%d-%s", totalPods, string(util.NewUUID()))

	// TODO: Use a more realistic workload
	Expect(RunRC(RCConfig{
		Client:    framework.Client,
		Name:      rcName,
		Namespace: framework.Namespace.Name,
		Image:     "gcr.io/google_containers/pause:go",
		Replicas:  totalPods,
	})).NotTo(HaveOccurred())

	// Log once and flush the stats.
	resourceMonitor.LogLatest()
	resourceMonitor.Reset()

	By("Start monitoring resource usage")
	// Periodically dump the cpu summary until the deadline is met.
	// Note that without calling resourceMonitor.Reset(), the stats
	// would occupy increasingly more memory. This should be fine
	// for the current test duration, but we should reclaim the
	// entries if we plan to monitor longer (e.g., 8 hours).
	deadline := time.Now().Add(monitoringTime)
	for time.Now().Before(deadline) {
		Logf("Still running...%v left", deadline.Sub(time.Now()))
		time.Sleep(reportingPeriod)
		timeLeft := deadline.Sub(time.Now())
		Logf("Still running...%v left", timeLeft)
		if timeLeft < reportingPeriod {
			time.Sleep(timeLeft)
		} else {
			time.Sleep(reportingPeriod)
		}
		logPodsOnNodes(framework.Client, nodeNames.List())
	}

	By("Reporting overall resource usage")
	logPodsOnNodes(framework.Client, nodeNames.List())
	resourceMonitor.LogCPUSummary()
	resourceMonitor.LogLatest()

	By("Deleting the RC")
	DeleteRC(framework.Client, framework.Namespace.Name, rcName)
}

var _ = Describe("Kubelet", func() {
	var nodeNames sets.String
	framework := NewFramework("kubelet-perf")
	var resourceMonitor *resourceMonitor

	BeforeEach(func() {
		nodes, err := framework.Client.Nodes().List(labels.Everything(), fields.Everything())
		expectNoError(err)
		nodeNames = sets.NewString()
		for _, node := range nodes.Items {
			nodeNames.Insert(node.Name)
		}
		resourceMonitor = newResourceMonitor(framework.Client, targetContainers(), containerStatsPollingPeriod)
		resourceMonitor.Start()
	})

	AfterEach(func() {
		resourceMonitor.Stop()
	})

	Describe("regular resource usage tracking", func() {
		density := []int{0, 35}
		for i := range density {
			podsPerNode := density[i]
			name := fmt.Sprintf(
				"over %v with %d pods per node.", monitoringTime, podsPerNode)
			It(name, func() {
				runResourceTrackingTest(framework, podsPerNode, nodeNames, resourceMonitor)
			})
		}
	})
	Describe("experimental resource usage tracking", func() {
		density := []int{50}
		for i := range density {
			podsPerNode := density[i]
			name := fmt.Sprintf(
				"over %v with %d pods per node.", monitoringTime, podsPerNode)
			It(name, func() {
				runResourceTrackingTest(framework, podsPerNode, nodeNames, resourceMonitor)
			})
		}
	})
})
