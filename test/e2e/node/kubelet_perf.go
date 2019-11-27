/*
Copyright 2015 The Kubernetes Authors.

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

package node

import (
	"fmt"
	"strings"
	"time"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	clientset "k8s.io/client-go/kubernetes"
	kubeletstatsv1alpha1 "k8s.io/kubernetes/pkg/kubelet/apis/stats/v1alpha1"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ekubelet "k8s.io/kubernetes/test/e2e/framework/kubelet"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2eperf "k8s.io/kubernetes/test/e2e/framework/perf"
	e2erc "k8s.io/kubernetes/test/e2e/framework/rc"
	"k8s.io/kubernetes/test/e2e/perftype"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
)

const (
	// Interval to poll /stats/container on a node
	containerStatsPollingPeriod = 10 * time.Second
	// The monitoring time for one test.
	monitoringTime = 20 * time.Minute
	// The periodic reporting period.
	reportingPeriod = 5 * time.Minute
)

type resourceTest struct {
	podsPerNode int
	cpuLimits   e2ekubelet.ContainersCPUSummary
	memLimits   e2ekubelet.ResourceUsagePerContainer
}

func logPodsOnNodes(c clientset.Interface, nodeNames []string) {
	for _, n := range nodeNames {
		podList, err := e2ekubelet.GetKubeletRunningPods(c, n)
		if err != nil {
			framework.Logf("Unable to retrieve kubelet pods for node %v", n)
			continue
		}
		framework.Logf("%d pods are running on node %v", len(podList.Items), n)
	}
}

func runResourceTrackingTest(f *framework.Framework, podsPerNode int, nodeNames sets.String, rm *e2ekubelet.ResourceMonitor,
	expectedCPU map[string]map[float64]float64, expectedMemory e2ekubelet.ResourceUsagePerContainer) {
	numNodes := nodeNames.Len()
	totalPods := podsPerNode * numNodes
	ginkgo.By(fmt.Sprintf("Creating a RC of %d pods and wait until all pods of this RC are running", totalPods))
	rcName := fmt.Sprintf("resource%d-%s", totalPods, string(uuid.NewUUID()))

	// TODO: Use a more realistic workload
	err := e2erc.RunRC(testutils.RCConfig{
		Client:    f.ClientSet,
		Name:      rcName,
		Namespace: f.Namespace.Name,
		Image:     imageutils.GetPauseImageName(),
		Replicas:  totalPods,
	})
	framework.ExpectNoError(err)

	// Log once and flush the stats.
	rm.LogLatest()
	rm.Reset()

	ginkgo.By("Start monitoring resource usage")
	// Periodically dump the cpu summary until the deadline is met.
	// Note that without calling e2ekubelet.ResourceMonitor.Reset(), the stats
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
		logPodsOnNodes(f.ClientSet, nodeNames.List())
	}

	ginkgo.By("Reporting overall resource usage")
	logPodsOnNodes(f.ClientSet, nodeNames.List())
	usageSummary, err := rm.GetLatest()
	framework.ExpectNoError(err)
	// TODO(random-liu): Remove the original log when we migrate to new perfdash
	framework.Logf("%s", rm.FormatResourceUsage(usageSummary))
	// Log perf result
	printPerfData(e2eperf.ResourceUsageToPerfData(rm.GetMasterNodeLatest(usageSummary)))
	verifyMemoryLimits(f.ClientSet, expectedMemory, usageSummary)

	cpuSummary := rm.GetCPUSummary()
	framework.Logf("%s", rm.FormatCPUSummary(cpuSummary))
	// Log perf result
	printPerfData(e2eperf.CPUUsageToPerfData(rm.GetMasterNodeCPUSummary(cpuSummary)))
	verifyCPULimits(expectedCPU, cpuSummary)

	ginkgo.By("Deleting the RC")
	e2erc.DeleteRCAndWaitForGC(f.ClientSet, f.Namespace.Name, rcName)
}

func verifyMemoryLimits(c clientset.Interface, expected e2ekubelet.ResourceUsagePerContainer, actual e2ekubelet.ResourceUsagePerNode) {
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
			heapStats, err := e2ekubelet.GetKubeletHeapStats(c, nodeName)
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

func verifyCPULimits(expected e2ekubelet.ContainersCPUSummary, actual e2ekubelet.NodesCPUSummary) {
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

// Slow by design (1 hour)
var _ = SIGDescribe("Kubelet [Serial] [Slow]", func() {
	var nodeNames sets.String
	f := framework.NewDefaultFramework("kubelet-perf")
	var om *e2ekubelet.RuntimeOperationMonitor
	var rm *e2ekubelet.ResourceMonitor

	ginkgo.BeforeEach(func() {
		nodes, err := e2enode.GetReadySchedulableNodes(f.ClientSet)
		framework.ExpectNoError(err)
		nodeNames = sets.NewString()
		for _, node := range nodes.Items {
			nodeNames.Insert(node.Name)
		}
		om = e2ekubelet.NewRuntimeOperationMonitor(f.ClientSet)
		rm = e2ekubelet.NewResourceMonitor(f.ClientSet, e2ekubelet.TargetContainers(), containerStatsPollingPeriod)
		rm.Start()
	})

	ginkgo.AfterEach(func() {
		rm.Stop()
		result := om.GetLatestRuntimeOperationErrorRate()
		framework.Logf("runtime operation error metrics:\n%s", e2ekubelet.FormatRuntimeOperationErrorRate(result))
	})
	SIGDescribe("regular resource usage tracking [Feature:RegularResourceUsageTracking]", func() {
		// We assume that the scheduler will make reasonable scheduling choices
		// and assign ~N pods on the node.
		// Although we want to track N pods per node, there are N + add-on pods
		// in the cluster. The cluster add-on pods can be distributed unevenly
		// among the nodes because they are created during the cluster
		// initialization. This *noise* is obvious when N is small. We
		// deliberately set higher resource usage limits to account for the
		// noise.
		//
		// We set all resource limits generously because this test is mainly
		// used to catch resource leaks in the soak cluster. For tracking
		// kubelet/runtime resource usage, please see the node e2e benchmark
		// dashboard. http://node-perf-dash.k8s.io/
		//
		// TODO(#36621): Deprecate this test once we have a node e2e soak
		// cluster.
		rTests := []resourceTest{
			{
				podsPerNode: 0,
				cpuLimits: e2ekubelet.ContainersCPUSummary{
					kubeletstatsv1alpha1.SystemContainerKubelet: {0.50: 0.10, 0.95: 0.20},
					kubeletstatsv1alpha1.SystemContainerRuntime: {0.50: 0.10, 0.95: 0.20},
				},
				memLimits: e2ekubelet.ResourceUsagePerContainer{
					kubeletstatsv1alpha1.SystemContainerKubelet: &e2ekubelet.ContainerResourceUsage{MemoryRSSInBytes: 200 * 1024 * 1024},
					// The detail can be found at https://github.com/kubernetes/kubernetes/issues/28384#issuecomment-244158892
					kubeletstatsv1alpha1.SystemContainerRuntime: &e2ekubelet.ContainerResourceUsage{MemoryRSSInBytes: 125 * 1024 * 1024},
				},
			},
			{
				cpuLimits: e2ekubelet.ContainersCPUSummary{
					kubeletstatsv1alpha1.SystemContainerKubelet: {0.50: 0.35, 0.95: 0.50},
					kubeletstatsv1alpha1.SystemContainerRuntime: {0.50: 0.10, 0.95: 0.50},
				},
				podsPerNode: 100,
				memLimits: e2ekubelet.ResourceUsagePerContainer{
					kubeletstatsv1alpha1.SystemContainerKubelet: &e2ekubelet.ContainerResourceUsage{MemoryRSSInBytes: 300 * 1024 * 1024},
					kubeletstatsv1alpha1.SystemContainerRuntime: &e2ekubelet.ContainerResourceUsage{MemoryRSSInBytes: 350 * 1024 * 1024},
				},
			},
		}
		for _, testArg := range rTests {
			itArg := testArg
			podsPerNode := itArg.podsPerNode
			name := fmt.Sprintf(
				"resource tracking for %d pods per node", podsPerNode)
			ginkgo.It(name, func() {
				runResourceTrackingTest(f, podsPerNode, nodeNames, rm, itArg.cpuLimits, itArg.memLimits)
			})
		}
	})
	SIGDescribe("experimental resource usage tracking [Feature:ExperimentalResourceUsageTracking]", func() {
		density := []int{100}
		for i := range density {
			podsPerNode := density[i]
			name := fmt.Sprintf(
				"resource tracking for %d pods per node", podsPerNode)
			ginkgo.It(name, func() {
				runResourceTrackingTest(f, podsPerNode, nodeNames, rm, nil, nil)
			})
		}
	})
})

// printPerfData prints the perfdata in json format with PerfResultTag prefix.
// If an error occurs, nothing will be printed.
func printPerfData(p *perftype.PerfData) {
	// Notice that we must make sure the perftype.PerfResultEnd is in a new line.
	if str := framework.PrettyPrintJSON(p); str != "" {
		framework.Logf("%s %s\n%s", perftype.PerfResultTag, str, perftype.PerfResultEnd)
	}
}
