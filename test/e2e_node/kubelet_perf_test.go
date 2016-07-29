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
	"fmt"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/stats"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("Kubelet-perf [Serial] [Slow]", func() {
	const (
		// Interval to poll /stats/container on a node
		containerStatsPollingPeriod = 10 * time.Second
		// The monitoring time for one test.
		monitoringTime = 6 * time.Minute
		// The periodic reporting period.
		reportingPeriod = 3 * time.Minute

		sleepAfterCreatePods = 10 * time.Second
		sleepAfterDeletePods = 120 * time.Second
	)

	var (
		ns string
		rm *ResourceCollector
		om *framework.RuntimeOperationMonitor
	)

	f := framework.NewDefaultFramework("kubelet-perf")

	BeforeEach(func() {
		ns = f.Namespace.Name
		om = framework.NewRuntimeOperationMonitor(f.Client)
	})

	AfterEach(func() {
		result := om.GetLatestRuntimeOperationErrorRate()
		framework.Logf("runtime operation error metrics:\n%s", framework.FormatRuntimeOperationErrorRate(result))
	})

	Context("regular resource usage tracking", func() {
		rTests := []resourceTest{
			{
				podsPerNode: 0,
				cpuLimits: framework.ContainersCPUSummary{
					stats.SystemContainerKubelet: {0.50: 0.06, 0.95: 0.08},
					stats.SystemContainerRuntime: {0.50: 0.05, 0.95: 0.06},
				},
				// We set the memory limits generously because the distribution
				// of the addon pods affect the memory usage on each node.
				memLimits: framework.ResourceUsagePerContainer{
					stats.SystemContainerKubelet: &framework.ContainerResourceUsage{MemoryRSSInBytes: 70 * 1024 * 1024},
					stats.SystemContainerRuntime: &framework.ContainerResourceUsage{MemoryRSSInBytes: 85 * 1024 * 1024},
				},
			},
			{
				podsPerNode: 35,
				cpuLimits: framework.ContainersCPUSummary{
					stats.SystemContainerKubelet: {0.50: 0.12, 0.95: 0.14},
					stats.SystemContainerRuntime: {0.50: 0.05, 0.95: 0.07},
				},
				// We set the memory limits generously because the distribution
				// of the addon pods affect the memory usage on each node.
				memLimits: framework.ResourceUsagePerContainer{
					stats.SystemContainerKubelet: &framework.ContainerResourceUsage{MemoryRSSInBytes: 70 * 1024 * 1024},
					stats.SystemContainerRuntime: &framework.ContainerResourceUsage{MemoryRSSInBytes: 150 * 1024 * 1024},
				},
			},
			{
				podsPerNode: 100,
				cpuLimits: framework.ContainersCPUSummary{
					stats.SystemContainerKubelet: {0.50: 0.17, 0.95: 0.22},
					stats.SystemContainerRuntime: {0.50: 0.06, 0.95: 0.09},
				},
				// We set the memory limits generously because the distribution
				// of the addon pods affect the memory usage on each node.
				memLimits: framework.ResourceUsagePerContainer{
					stats.SystemContainerKubelet: &framework.ContainerResourceUsage{MemoryRSSInBytes: 80 * 1024 * 1024},
					stats.SystemContainerRuntime: &framework.ContainerResourceUsage{MemoryRSSInBytes: 300 * 1024 * 1024},
				},
			},
		}

		for _, testArg := range rTests {
			itArg := testArg

			podsPerNode := itArg.podsPerNode
			name := fmt.Sprintf("resource tracking for %d pods per node", podsPerNode)

			It(name, func() {
				expectedCPU, expectedMemory := itArg.cpuLimits, itArg.memLimits

				createCadvisorPod(f)
				rm = NewResourceCollector(containerStatsPollingPeriod)
				rm.Start()

				By("Creating a batch of Pods")
				pods := newTestPods(podsPerNode, ImageRegistry[pauseImage], "test_pod")
				for _, pod := range pods {
					f.PodClient().CreateSync(pod)
				}

				// wait for a while to let the node be steady
				time.Sleep(sleepAfterCreatePods)

				// Log once and flush the stats.
				rm.LogLatest()
				rm.Reset()

				By("Start monitoring resource usage")
				// Periodically dump the cpu summary until the deadline is met.
				// Note that without calling framework.ResourceMonitor.Reset(), the stats
				// would occupy increasingly more memory. This should be fine
				// for the current test duration, but we should reclaim the
				// entries if we plan to monitor longer (e.g., 8 hours).
				deadline := time.Now().Add(monitoringTime)
				for time.Now().Before(deadline) {
					timeLeft := deadline.Sub(time.Now())
					framework.Logf("Still running...%v left", timeLeft)
					if timeLeft < reportingPeriod {
						time.Sleep(timeLeft)
					} else {
						time.Sleep(reportingPeriod)
					}
					logPodsOnNodes(f.Client)
				}

				By("Reporting overall resource usage")
				logPodsOnNodes(f.Client)

				usagePerContainer, err := rm.GetLatest()
				Expect(err).NotTo(HaveOccurred())

				// TODO(random-liu): Remove the original log when we migrate to new perfdash
				nodeName := framework.TestContext.NodeName
				framework.Logf("%s", formatResourceUsageStats(usagePerContainer))

				// Log perf result
				usagePerNode := make(framework.ResourceUsagePerNode)
				usagePerNode[nodeName] = usagePerContainer

				framework.PrintPerfData(framework.ResourceUsageToPerfData(usagePerNode))
				verifyMemoryLimits(f.Client, expectedMemory, usagePerNode)

				cpuSummary := rm.GetCPUSummary()
				framework.Logf("%s", formatCPUSummary(cpuSummary))

				// Log perf result
				cpuSummaryPerNode := make(framework.NodesCPUSummary)
				cpuSummaryPerNode[nodeName] = cpuSummary
				framework.PrintPerfData(framework.CPUUsageToPerfData(cpuSummaryPerNode))
				verifyCPULimits(expectedCPU, cpuSummaryPerNode)

				// delete pods
				By("Deleting a batch of pods")
				deleteBatchPod(f, pods)

				rm.Stop()

				// tear down cadvisor
				Expect(f.Client.Pods(ns).Delete(cadvisorPodName, api.NewDeleteOptions(30))).
					NotTo(HaveOccurred())
				Expect(framework.WaitForPodToDisappear(f.Client, ns, cadvisorPodName, labels.Everything(),
					3*time.Second, 10*time.Minute)).
					NotTo(HaveOccurred())

				time.Sleep(sleepAfterDeletePods)
			})
		}
	})
})

type resourceTest struct {
	podsPerNode int
	cpuLimits   framework.ContainersCPUSummary
	memLimits   framework.ResourceUsagePerContainer
}

func verifyMemoryLimits(c *client.Client, expected framework.ResourceUsagePerContainer, actual framework.ResourceUsagePerNode) {
	if expected == nil {
		return
	}
	var errList []string
	for nodeName, nodeSummary := range actual {
		var nodeErrs []string
		for cName, expectedResult := range expected {
			container, ok := nodeSummary[cName]
			if !ok {
				nodeErrs = append(nodeErrs, fmt.Sprintf("container %q: missing", cName))
				continue
			}

			expectedValue := expectedResult.MemoryRSSInBytes
			actualValue := container.MemoryRSSInBytes
			if expectedValue != 0 && actualValue > expectedValue {
				nodeErrs = append(nodeErrs, fmt.Sprintf("container %q: expected RSS memory (MB) < %d; got %d",
					cName, expectedValue, actualValue))
			}
		}
		if len(nodeErrs) > 0 {
			errList = append(errList, fmt.Sprintf("node %v:\n %s", nodeName, strings.Join(nodeErrs, ", ")))
			heapStats, err := framework.GetKubeletHeapStats(c, nodeName)
			if err != nil {
				framework.Logf("Unable to get heap stats from %q", nodeName)
			} else {
				framework.Logf("Heap stats on %q\n:%v", nodeName, heapStats)
			}
		}
	}
	if len(errList) > 0 {
		framework.Failf("Memory usage exceeding limits:\n %s", strings.Join(errList, "\n"))
	}
}

func verifyCPULimits(expected framework.ContainersCPUSummary, actual framework.NodesCPUSummary) {
	if expected == nil {
		return
	}
	var errList []string
	for nodeName, perNodeSummary := range actual {
		var nodeErrs []string
		for cName, expectedResult := range expected {
			perContainerSummary, ok := perNodeSummary[cName]
			if !ok {
				nodeErrs = append(nodeErrs, fmt.Sprintf("container %q: missing", cName))
				continue
			}
			for p, expectedValue := range expectedResult {
				actualValue, ok := perContainerSummary[p]
				if !ok {
					nodeErrs = append(nodeErrs, fmt.Sprintf("container %q: missing percentile %v", cName, p))
					continue
				}
				if actualValue > expectedValue {
					nodeErrs = append(nodeErrs, fmt.Sprintf("container %q: expected %.0fth%% usage < %.3f; got %.3f",
						cName, p*100, expectedValue, actualValue))
				}
			}
		}
		if len(nodeErrs) > 0 {
			errList = append(errList, fmt.Sprintf("node %v:\n %s", nodeName, strings.Join(nodeErrs, ", ")))
		}
	}
	if len(errList) > 0 {
		framework.Failf("CPU usage exceeding limits:\n %s", strings.Join(errList, "\n"))
	}
}

func logPodsOnNodes(c *client.Client) {
	nodeName := framework.TestContext.NodeName
	podList, err := framework.GetKubeletRunningPods(c, nodeName)
	if err != nil {
		framework.Logf("Unable to retrieve kubelet pods for node %v", nodeName)
	}
	framework.Logf("%d pods are running on node %v", len(podList.Items), nodeName)
}
