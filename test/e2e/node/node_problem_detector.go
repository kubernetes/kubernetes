/*
Copyright 2019 The Kubernetes Authors.

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
	"sort"
	"strconv"
	"strings"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/kubernetes/test/e2e/framework"
	testutils "k8s.io/kubernetes/test/utils"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// This test checks if node-problem-detector (NPD) runs fine without error on
// the nodes in the cluster. NPD's functionality is tested in e2e_node tests.
var _ = SIGDescribe("NodeProblemDetector", func() {
	const (
		pollInterval = 1 * time.Second
		pollTimeout  = 1 * time.Minute
	)
	f := framework.NewDefaultFramework("node-problem-detector")

	BeforeEach(func() {
		framework.SkipUnlessSSHKeyPresent()
		framework.SkipUnlessProviderIs(framework.ProvidersWithSSH...)
		framework.SkipUnlessProviderIs("gce", "gke")
		framework.SkipUnlessNodeOSDistroIs("gci", "ubuntu")
		framework.WaitForAllNodesHealthy(f.ClientSet, time.Minute)
	})

	It("should run without error", func() {
		By("Getting all nodes' SSH-able IP addresses")
		hosts, err := framework.NodeSSHHosts(f.ClientSet)
		if err != nil {
			framework.Failf("Error getting node hostnames: %v", err)
		}
		Expect(len(hosts)).NotTo(BeZero())

		cpuUsageStats := make(map[string][]float64)
		uptimeStats := make(map[string][]float64)
		rssStats := make(map[string][]float64)
		workingSetStats := make(map[string][]float64)

		for _, host := range hosts {
			cpuUsageStats[host] = []float64{}
			uptimeStats[host] = []float64{}
			rssStats[host] = []float64{}
			workingSetStats[host] = []float64{}

			By(fmt.Sprintf("Check node %q has node-problem-detector process", host))
			psCmd := "ps aux | grep node-problem-detector"
			result, err := framework.SSH(psCmd, host, framework.TestContext.Provider)
			framework.ExpectNoError(err)
			Expect(result.Code).To(BeZero())
			Expect(result.Stdout).To(ContainSubstring("/home/kubernetes/bin/node-problem-detector"))

			By(fmt.Sprintf("Check node-problem-detector is running fine on node %q", host))
			journalctlCmd := "sudo journalctl -u node-problem-detector"
			result, err = framework.SSH(journalctlCmd, host, framework.TestContext.Provider)
			framework.ExpectNoError(err)
			Expect(result.Code).To(BeZero())
			Expect(result.Stdout).NotTo(ContainSubstring("node-problem-detector.service: Failed"))

			cpuUsage, uptime := getCpuStat(f, host)
			cpuUsageStats[host] = append(cpuUsageStats[host], cpuUsage)
			uptimeStats[host] = append(uptimeStats[host], uptime)

			By(fmt.Sprintf("Inject log to trigger AUFSUmountHung on node %q", host))
			log := "INFO: task umount.aufs:21568 blocked for more than 120 seconds."
			injectLogCmd := "sudo sh -c \"echo 'kernel: " + log + "' >> /dev/kmsg\""
			_, err = framework.SSH(injectLogCmd, host, framework.TestContext.Provider)
			framework.ExpectNoError(err)
			Expect(result.Code).To(BeZero())
		}

		By("Check node-problem-detector can post conditions and events to API server")
		nodes := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
		Expect(len(nodes.Items)).To(Equal(len(hosts)))
		for _, node := range nodes.Items {
			By(fmt.Sprintf("Check node-problem-detector posted KernelDeadlock condition on node %q", node.Name))
			Eventually(func() error {
				return verifyNodeCondition(f, "KernelDeadlock", v1.ConditionTrue, "AUFSUmountHung", node.Name)
			}, pollTimeout, pollInterval).Should(Succeed())

			By(fmt.Sprintf("Check node-problem-detector posted AUFSUmountHung event on node %q", node.Name))
			eventListOptions := metav1.ListOptions{FieldSelector: fields.Set{"involvedObject.kind": "Node"}.AsSelector().String()}
			Eventually(func() error {
				return verifyEvents(f, eventListOptions, 1, "AUFSUmountHung", node.Name)
			}, pollTimeout, pollInterval).Should(Succeed())
		}

		By("Gather node-problem-detector cpu and memory stats")
		numIterations := 60
		for i := 1; i <= numIterations; i++ {
			for _, host := range hosts {
				rss, workingSet := getMemoryStat(f, host)
				rssStats[host] = append(rssStats[host], rss)
				workingSetStats[host] = append(workingSetStats[host], workingSet)
				if i == numIterations {
					cpuUsage, uptime := getCpuStat(f, host)
					cpuUsageStats[host] = append(cpuUsageStats[host], cpuUsage)
					uptimeStats[host] = append(uptimeStats[host], uptime)
				}
			}
			time.Sleep(time.Second)
		}

		cpuStatsMsg := "CPU (core):"
		rssStatsMsg := "RSS (MB):"
		workingSetStatsMsg := "WorkingSet (MB):"
		for i, host := range hosts {
			cpuUsage := cpuUsageStats[host][1] - cpuUsageStats[host][0]
			totaltime := uptimeStats[host][1] - uptimeStats[host][0]
			cpuStatsMsg += fmt.Sprintf(" Node%d[%.3f];", i, cpuUsage/totaltime)

			sort.Float64s(rssStats[host])
			rssStatsMsg += fmt.Sprintf(" Node%d[%.1f|%.1f|%.1f];", i,
				rssStats[host][0], rssStats[host][len(rssStats[host])/2], rssStats[host][len(rssStats[host])-1])

			sort.Float64s(workingSetStats[host])
			workingSetStatsMsg += fmt.Sprintf(" Node%d[%.1f|%.1f|%.1f];", i,
				workingSetStats[host][0], workingSetStats[host][len(workingSetStats[host])/2], workingSetStats[host][len(workingSetStats[host])-1])
		}
		framework.Logf("Node-Problem-Detector CPU and Memory Stats:\n\t%s\n\t%s\n\t%s", cpuStatsMsg, rssStatsMsg, workingSetStatsMsg)
	})
})

func verifyEvents(f *framework.Framework, options metav1.ListOptions, num int, reason, nodeName string) error {
	events, err := f.ClientSet.CoreV1().Events(metav1.NamespaceDefault).List(options)
	if err != nil {
		return err
	}
	count := 0
	for _, event := range events.Items {
		if event.Reason != reason || event.Source.Host != nodeName {
			continue
		}
		count += int(event.Count)
	}
	if count != num {
		return fmt.Errorf("expect event number %d, got %d: %v", num, count, events.Items)
	}
	return nil
}

func verifyNodeCondition(f *framework.Framework, condition v1.NodeConditionType, status v1.ConditionStatus, reason, nodeName string) error {
	node, err := f.ClientSet.CoreV1().Nodes().Get(nodeName, metav1.GetOptions{})
	if err != nil {
		return err
	}
	_, c := testutils.GetNodeCondition(&node.Status, condition)
	if c == nil {
		return fmt.Errorf("node condition %q not found", condition)
	}
	if c.Status != status || c.Reason != reason {
		return fmt.Errorf("unexpected node condition %q: %+v", condition, c)
	}
	return nil
}

func getMemoryStat(f *framework.Framework, host string) (rss, workingSet float64) {
	memCmd := "cat /sys/fs/cgroup/memory/system.slice/node-problem-detector.service/memory.usage_in_bytes && cat /sys/fs/cgroup/memory/system.slice/node-problem-detector.service/memory.stat"
	result, err := framework.SSH(memCmd, host, framework.TestContext.Provider)
	framework.ExpectNoError(err)
	Expect(result.Code).To(BeZero())
	lines := strings.Split(result.Stdout, "\n")

	memoryUsage, err := strconv.ParseFloat(lines[0], 64)
	Expect(err).To(BeNil())

	var totalInactiveFile float64
	for _, line := range lines[1:] {
		tokens := strings.Split(line, " ")
		if tokens[0] == "total_rss" {
			rss, err = strconv.ParseFloat(tokens[1], 64)
			Expect(err).To(BeNil())
		}
		if tokens[0] == "total_inactive_file" {
			totalInactiveFile, err = strconv.ParseFloat(tokens[1], 64)
			Expect(err).To(BeNil())
		}
	}

	workingSet = memoryUsage
	if workingSet < totalInactiveFile {
		workingSet = 0
	} else {
		workingSet -= totalInactiveFile
	}

	// Convert to MB
	rss = rss / 1024 / 1024
	workingSet = workingSet / 1024 / 1024
	return
}

func getCpuStat(f *framework.Framework, host string) (usage, uptime float64) {
	cpuCmd := "cat /sys/fs/cgroup/cpu/system.slice/node-problem-detector.service/cpuacct.usage && cat /proc/uptime | awk '{print $1}'"
	result, err := framework.SSH(cpuCmd, host, framework.TestContext.Provider)
	framework.ExpectNoError(err)
	Expect(result.Code).To(BeZero())
	lines := strings.Split(result.Stdout, "\n")

	usage, err = strconv.ParseFloat(lines[0], 64)
	uptime, err = strconv.ParseFloat(lines[1], 64)

	// Convert from nanoseconds to seconds
	usage *= 1e-9
	return
}
