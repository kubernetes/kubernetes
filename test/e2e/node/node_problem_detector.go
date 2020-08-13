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
	"context"
	"fmt"
	"net"
	"sort"
	"strconv"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ekubelet "k8s.io/kubernetes/test/e2e/framework/kubelet"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2essh "k8s.io/kubernetes/test/e2e/framework/ssh"
	testutils "k8s.io/kubernetes/test/utils"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

// This test checks if node-problem-detector (NPD) runs fine without error on
// the nodes in the cluster. NPD's functionality is tested in e2e_node tests.
var _ = SIGDescribe("NodeProblemDetector [DisabledForLargeClusters]", func() {
	const (
		pollInterval = 1 * time.Second
		pollTimeout  = 1 * time.Minute
	)
	f := framework.NewDefaultFramework("node-problem-detector")

	ginkgo.BeforeEach(func() {
		e2eskipper.SkipUnlessSSHKeyPresent()
		e2eskipper.SkipUnlessProviderIs(framework.ProvidersWithSSH...)
		e2eskipper.SkipUnlessProviderIs("gce", "gke")
		e2eskipper.SkipUnlessNodeOSDistroIs("gci", "ubuntu")
		e2enode.WaitForTotalHealthy(f.ClientSet, time.Minute)
	})

	ginkgo.It("should run without error", func() {
		e2eskipper.SkipUnlessSSHKeyPresent()

		ginkgo.By("Getting all nodes and their SSH-able IP addresses")
		nodes, err := e2enode.GetReadySchedulableNodes(f.ClientSet)
		framework.ExpectNoError(err)
		hosts := []string{}
		for _, node := range nodes.Items {
			for _, addr := range node.Status.Addresses {
				if addr.Type == v1.NodeExternalIP {
					hosts = append(hosts, net.JoinHostPort(addr.Address, "22"))
					break
				}
			}
		}
		framework.ExpectEqual(len(hosts), len(nodes.Items))

		isStandaloneMode := make(map[string]bool)
		cpuUsageStats := make(map[string][]float64)
		uptimeStats := make(map[string][]float64)
		rssStats := make(map[string][]float64)
		workingSetStats := make(map[string][]float64)

		for _, host := range hosts {
			cpuUsageStats[host] = []float64{}
			uptimeStats[host] = []float64{}
			rssStats[host] = []float64{}
			workingSetStats[host] = []float64{}

			cmd := "systemctl status node-problem-detector.service"
			result, err := e2essh.SSH(cmd, host, framework.TestContext.Provider)
			isStandaloneMode[host] = (err == nil && result.Code == 0)

			ginkgo.By(fmt.Sprintf("Check node %q has node-problem-detector process", host))
			// Using brackets "[n]" is a trick to prevent grep command itself from
			// showing up, because string text "[n]ode-problem-detector" does not
			// match regular expression "[n]ode-problem-detector".
			psCmd := "ps aux | grep [n]ode-problem-detector"
			result, err = e2essh.SSH(psCmd, host, framework.TestContext.Provider)
			framework.ExpectNoError(err)
			framework.ExpectEqual(result.Code, 0)
			gomega.Expect(result.Stdout).To(gomega.ContainSubstring("node-problem-detector"))

			ginkgo.By(fmt.Sprintf("Check node-problem-detector is running fine on node %q", host))
			journalctlCmd := "sudo journalctl -u node-problem-detector"
			result, err = e2essh.SSH(journalctlCmd, host, framework.TestContext.Provider)
			framework.ExpectNoError(err)
			framework.ExpectEqual(result.Code, 0)
			gomega.Expect(result.Stdout).NotTo(gomega.ContainSubstring("node-problem-detector.service: Failed"))

			if isStandaloneMode[host] {
				cpuUsage, uptime := getCPUStat(f, host)
				cpuUsageStats[host] = append(cpuUsageStats[host], cpuUsage)
				uptimeStats[host] = append(uptimeStats[host], uptime)
			}

			ginkgo.By(fmt.Sprintf("Inject log to trigger AUFSUmountHung on node %q", host))
			log := "INFO: task umount.aufs:21568 blocked for more than 120 seconds."
			injectLogCmd := "sudo sh -c \"echo 'kernel: " + log + "' >> /dev/kmsg\""
			_, err = e2essh.SSH(injectLogCmd, host, framework.TestContext.Provider)
			framework.ExpectNoError(err)
			framework.ExpectEqual(result.Code, 0)
		}

		ginkgo.By("Check node-problem-detector can post conditions and events to API server")
		for _, node := range nodes.Items {
			ginkgo.By(fmt.Sprintf("Check node-problem-detector posted KernelDeadlock condition on node %q", node.Name))
			gomega.Eventually(func() error {
				return verifyNodeCondition(f, "KernelDeadlock", v1.ConditionTrue, "AUFSUmountHung", node.Name)
			}, pollTimeout, pollInterval).Should(gomega.Succeed())

			ginkgo.By(fmt.Sprintf("Check node-problem-detector posted AUFSUmountHung event on node %q", node.Name))
			eventListOptions := metav1.ListOptions{FieldSelector: fields.Set{"involvedObject.kind": "Node"}.AsSelector().String()}
			gomega.Eventually(func() error {
				return verifyEvents(f, eventListOptions, 1, "AUFSUmountHung", node.Name)
			}, pollTimeout, pollInterval).Should(gomega.Succeed())

			// Node problem detector reports kubelet start events automatically starting from NPD v0.7.0+.
			// Since Kubelet may be restarted for a few times after node is booted. We just check the event
			// is detected, but do not check how many times Kubelet is started.
			ginkgo.By(fmt.Sprintf("Check node-problem-detector posted KubeletStart event on node %q", node.Name))
			gomega.Eventually(func() error {
				return verifyEventExists(f, eventListOptions, "KubeletStart", node.Name)
			}, pollTimeout, pollInterval).Should(gomega.Succeed())
		}

		ginkgo.By("Gather node-problem-detector cpu and memory stats")
		numIterations := 60
		for i := 1; i <= numIterations; i++ {
			for j, host := range hosts {
				if isStandaloneMode[host] {
					rss, workingSet := getMemoryStat(f, host)
					rssStats[host] = append(rssStats[host], rss)
					workingSetStats[host] = append(workingSetStats[host], workingSet)
					if i == numIterations {
						cpuUsage, uptime := getCPUStat(f, host)
						cpuUsageStats[host] = append(cpuUsageStats[host], cpuUsage)
						uptimeStats[host] = append(uptimeStats[host], uptime)
					}
				} else {
					cpuUsage, rss, workingSet := getNpdPodStat(f, nodes.Items[j].Name)
					cpuUsageStats[host] = append(cpuUsageStats[host], cpuUsage)
					rssStats[host] = append(rssStats[host], rss)
					workingSetStats[host] = append(workingSetStats[host], workingSet)
				}
			}
			time.Sleep(time.Second)
		}

		cpuStatsMsg := "CPU (core):"
		rssStatsMsg := "RSS (MB):"
		workingSetStatsMsg := "WorkingSet (MB):"
		for i, host := range hosts {
			if isStandaloneMode[host] {
				// When in standalone mode, NPD is running as systemd service. We
				// calculate its cpu usage from cgroup cpuacct value differences.
				cpuUsage := cpuUsageStats[host][1] - cpuUsageStats[host][0]
				totaltime := uptimeStats[host][1] - uptimeStats[host][0]
				cpuStatsMsg += fmt.Sprintf(" %s[%.3f];", nodes.Items[i].Name, cpuUsage/totaltime)
			} else {
				sort.Float64s(cpuUsageStats[host])
				cpuStatsMsg += fmt.Sprintf(" %s[%.3f|%.3f|%.3f];", nodes.Items[i].Name,
					cpuUsageStats[host][0], cpuUsageStats[host][len(cpuUsageStats[host])/2], cpuUsageStats[host][len(cpuUsageStats[host])-1])
			}

			sort.Float64s(rssStats[host])
			rssStatsMsg += fmt.Sprintf(" %s[%.1f|%.1f|%.1f];", nodes.Items[i].Name,
				rssStats[host][0], rssStats[host][len(rssStats[host])/2], rssStats[host][len(rssStats[host])-1])

			sort.Float64s(workingSetStats[host])
			workingSetStatsMsg += fmt.Sprintf(" %s[%.1f|%.1f|%.1f];", nodes.Items[i].Name,
				workingSetStats[host][0], workingSetStats[host][len(workingSetStats[host])/2], workingSetStats[host][len(workingSetStats[host])-1])
		}
		framework.Logf("Node-Problem-Detector CPU and Memory Stats:\n\t%s\n\t%s\n\t%s", cpuStatsMsg, rssStatsMsg, workingSetStatsMsg)
	})
})

func verifyEvents(f *framework.Framework, options metav1.ListOptions, num int, reason, nodeName string) error {
	events, err := f.ClientSet.CoreV1().Events(metav1.NamespaceDefault).List(context.TODO(), options)
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

func verifyEventExists(f *framework.Framework, options metav1.ListOptions, reason, nodeName string) error {
	events, err := f.ClientSet.CoreV1().Events(metav1.NamespaceDefault).List(context.TODO(), options)
	if err != nil {
		return err
	}
	for _, event := range events.Items {
		if event.Reason == reason && event.Source.Host == nodeName && event.Count > 0 {
			return nil
		}
	}
	return fmt.Errorf("Event %s does not exist: %v", reason, events.Items)
}

func verifyNodeCondition(f *framework.Framework, condition v1.NodeConditionType, status v1.ConditionStatus, reason, nodeName string) error {
	node, err := f.ClientSet.CoreV1().Nodes().Get(context.TODO(), nodeName, metav1.GetOptions{})
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
	result, err := e2essh.SSH(memCmd, host, framework.TestContext.Provider)
	framework.ExpectNoError(err)
	framework.ExpectEqual(result.Code, 0)
	lines := strings.Split(result.Stdout, "\n")

	memoryUsage, err := strconv.ParseFloat(lines[0], 64)
	framework.ExpectNoError(err)

	var totalInactiveFile float64
	for _, line := range lines[1:] {
		tokens := strings.Split(line, " ")
		if tokens[0] == "total_rss" {
			rss, err = strconv.ParseFloat(tokens[1], 64)
			framework.ExpectNoError(err)
		}
		if tokens[0] == "total_inactive_file" {
			totalInactiveFile, err = strconv.ParseFloat(tokens[1], 64)
			framework.ExpectNoError(err)
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

func getCPUStat(f *framework.Framework, host string) (usage, uptime float64) {
	cpuCmd := "cat /sys/fs/cgroup/cpu/system.slice/node-problem-detector.service/cpuacct.usage && cat /proc/uptime | awk '{print $1}'"
	result, err := e2essh.SSH(cpuCmd, host, framework.TestContext.Provider)
	framework.ExpectNoError(err)
	framework.ExpectEqual(result.Code, 0)
	lines := strings.Split(result.Stdout, "\n")

	usage, err = strconv.ParseFloat(lines[0], 64)
	framework.ExpectNoError(err, "Cannot parse float for usage")
	uptime, err = strconv.ParseFloat(lines[1], 64)
	framework.ExpectNoError(err, "Cannot parse float for uptime")

	// Convert from nanoseconds to seconds
	usage *= 1e-9
	return
}

func getNpdPodStat(f *framework.Framework, nodeName string) (cpuUsage, rss, workingSet float64) {
	summary, err := e2ekubelet.GetStatsSummary(f.ClientSet, nodeName)
	framework.ExpectNoError(err)

	hasNpdPod := false
	for _, pod := range summary.Pods {
		if !strings.HasPrefix(pod.PodRef.Name, "npd") {
			continue
		}
		cpuUsage = float64(*pod.CPU.UsageNanoCores) * 1e-9
		rss = float64(*pod.Memory.RSSBytes) / 1024 / 1024
		workingSet = float64(*pod.Memory.WorkingSetBytes) / 1024 / 1024
		hasNpdPod = true
		break
	}
	framework.ExpectEqual(hasNpdPod, true)
	return
}
