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

package lifecycle

import (
	"fmt"
	"strings"
	"sync"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/test/e2e/framework"
	testutils "k8s.io/kubernetes/test/utils"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	// How long a node is allowed to go from "Ready" to "NotReady" after a
	// reboot is issued before the test is considered failed.
	rebootNodeNotReadyTimeout = 2 * time.Minute

	// How long a node is allowed to go from "NotReady" to "Ready" after a
	// reboot is issued and it is found to be "NotReady" before the test is
	// considered failed.
	rebootNodeReadyAgainTimeout = 5 * time.Minute

	// How long pods have to be "ready" after the reboot.
	rebootPodReadyAgainTimeout = 5 * time.Minute
)

var _ = SIGDescribe("Reboot [Disruptive] [Feature:Reboot]", func() {
	var f *framework.Framework

	BeforeEach(func() {
		// These tests requires SSH to nodes, so the provider check should be identical to there
		// (the limiting factor is the implementation of util.go's framework.GetSigner(...)).

		// Cluster must support node reboot
		framework.SkipUnlessProviderIs(framework.ProvidersWithSSH...)
	})

	AfterEach(func() {
		if CurrentGinkgoTestDescription().Failed {
			// Most of the reboot tests just make sure that addon/system pods are running, so dump
			// events for the kube-system namespace on failures
			namespaceName := metav1.NamespaceSystem
			By(fmt.Sprintf("Collecting events from namespace %q.", namespaceName))
			events, err := f.ClientSet.Core().Events(namespaceName).List(metav1.ListOptions{})
			Expect(err).NotTo(HaveOccurred())

			for _, e := range events.Items {
				framework.Logf("event for %v: %v %v: %v", e.InvolvedObject.Name, e.Source, e.Reason, e.Message)
			}
		}
		// In GKE, our current tunneling setup has the potential to hold on to a broken tunnel (from a
		// rebooted/deleted node) for up to 5 minutes before all tunnels are dropped and recreated.  Most tests
		// make use of some proxy feature to verify functionality. So, if a reboot test runs right before a test
		// that tries to get logs, for example, we may get unlucky and try to use a closed tunnel to a node that
		// was recently rebooted. There's no good way to framework.Poll for proxies being closed, so we sleep.
		//
		// TODO(cjcullen) reduce this sleep (#19314)
		if framework.ProviderIs("gke") {
			By("waiting 5 minutes for all dead tunnels to be dropped")
			time.Sleep(5 * time.Minute)
		}
	})

	f = framework.NewDefaultFramework("reboot")

	It("each node by ordering clean reboot and ensure they function upon restart", func() {
		// clean shutdown and restart
		// We sleep 10 seconds to give some time for ssh command to cleanly finish before the node is rebooted.
		testReboot(f.ClientSet, "nohup sh -c 'sleep 10 && sudo reboot' >/dev/null 2>&1 &", nil)
	})

	It("each node by ordering unclean reboot and ensure they function upon restart", func() {
		// unclean shutdown and restart
		// We sleep 10 seconds to give some time for ssh command to cleanly finish before the node is shutdown.
		testReboot(f.ClientSet, "nohup sh -c 'sleep 10 && echo b | sudo tee /proc/sysrq-trigger' >/dev/null 2>&1 &", nil)
	})

	It("each node by triggering kernel panic and ensure they function upon restart", func() {
		// kernel panic
		// We sleep 10 seconds to give some time for ssh command to cleanly finish before kernel panic is triggered.
		testReboot(f.ClientSet, "nohup sh -c 'sleep 10 && echo c | sudo tee /proc/sysrq-trigger' >/dev/null 2>&1 &", nil)
	})

	It("each node by switching off the network interface and ensure they function upon switch on", func() {
		// switch the network interface off for a while to simulate a network outage
		// We sleep 10 seconds to give some time for ssh command to cleanly finish before network is down.
		testReboot(f.ClientSet, "nohup sh -c 'sleep 10 && (sudo ifdown eth0 || sudo ip link set eth0 down) && sleep 120 && (sudo ifup eth0 || sudo ip link set eth0 up)' >/dev/null 2>&1 &", nil)
	})

	It("each node by dropping all inbound packets for a while and ensure they function afterwards", func() {
		// tell the firewall to drop all inbound packets for a while
		// We sleep 10 seconds to give some time for ssh command to cleanly finish before starting dropping inbound packets.
		// We still accept packages send from localhost to prevent monit from restarting kubelet.
		tmpLogPath := "/tmp/drop-inbound.log"
		testReboot(f.ClientSet, dropPacketsScript("INPUT", tmpLogPath), catLogHook(tmpLogPath))
	})

	It("each node by dropping all outbound packets for a while and ensure they function afterwards", func() {
		// tell the firewall to drop all outbound packets for a while
		// We sleep 10 seconds to give some time for ssh command to cleanly finish before starting dropping outbound packets.
		// We still accept packages send to localhost to prevent monit from restarting kubelet.
		tmpLogPath := "/tmp/drop-outbound.log"
		testReboot(f.ClientSet, dropPacketsScript("OUTPUT", tmpLogPath), catLogHook(tmpLogPath))
	})
})

func testReboot(c clientset.Interface, rebootCmd string, hook terminationHook) {
	// Get all nodes, and kick off the test on each.
	nodelist := framework.GetReadySchedulableNodesOrDie(c)
	if hook != nil {
		defer func() {
			framework.Logf("Executing termination hook on nodes")
			hook(framework.TestContext.Provider, nodelist)
		}()
	}
	result := make([]bool, len(nodelist.Items))
	wg := sync.WaitGroup{}
	wg.Add(len(nodelist.Items))

	failed := false
	for ix := range nodelist.Items {
		go func(ix int) {
			defer wg.Done()
			n := nodelist.Items[ix]
			result[ix] = rebootNode(c, framework.TestContext.Provider, n.ObjectMeta.Name, rebootCmd)
			if !result[ix] {
				failed = true
			}
		}(ix)
	}

	// Wait for all to finish and check the final result.
	wg.Wait()

	if failed {
		for ix := range nodelist.Items {
			n := nodelist.Items[ix]
			if !result[ix] {
				framework.Logf("Node %s failed reboot test.", n.ObjectMeta.Name)
			}
		}
		framework.Failf("Test failed; at least one node failed to reboot in the time given.")
	}
}

func printStatusAndLogsForNotReadyPods(c clientset.Interface, ns string, podNames []string, pods []*v1.Pod) {
	printFn := func(id, log string, err error, previous bool) {
		prefix := "Retrieving log for container"
		if previous {
			prefix = "Retrieving log for the last terminated container"
		}
		if err != nil {
			framework.Logf("%s %s, err: %v:\n%s\n", prefix, id, err, log)
		} else {
			framework.Logf("%s %s:\n%s\n", prefix, id, log)
		}
	}
	podNameSet := sets.NewString(podNames...)
	for _, p := range pods {
		if p.Namespace != ns {
			continue
		}
		if !podNameSet.Has(p.Name) {
			continue
		}
		if ok, _ := testutils.PodRunningReady(p); ok {
			continue
		}
		framework.Logf("Status for not ready pod %s/%s: %+v", p.Namespace, p.Name, p.Status)
		// Print the log of the containers if pod is not running and ready.
		for _, container := range p.Status.ContainerStatuses {
			cIdentifer := fmt.Sprintf("%s/%s/%s", p.Namespace, p.Name, container.Name)
			log, err := framework.GetPodLogs(c, p.Namespace, p.Name, container.Name)
			printFn(cIdentifer, log, err, false)
			// Get log from the previous container.
			if container.RestartCount > 0 {
				printFn(cIdentifer, log, err, true)
			}
		}
	}
}

// rebootNode takes node name on provider through the following steps using c:
//  - ensures the node is ready
//  - ensures all pods on the node are running and ready
//  - reboots the node (by executing rebootCmd over ssh)
//  - ensures the node reaches some non-ready state
//  - ensures the node becomes ready again
//  - ensures all pods on the node become running and ready again
//
// It returns true through result only if all of the steps pass; at the first
// failed step, it will return false through result and not run the rest.
func rebootNode(c clientset.Interface, provider, name, rebootCmd string) bool {
	// Setup
	ns := metav1.NamespaceSystem
	ps := testutils.NewPodStore(c, ns, labels.Everything(), fields.OneTermEqualSelector(api.PodHostField, name))
	defer ps.Stop()

	// Get the node initially.
	framework.Logf("Getting %s", name)
	node, err := c.Core().Nodes().Get(name, metav1.GetOptions{})
	if err != nil {
		framework.Logf("Couldn't get node %s", name)
		return false
	}

	// Node sanity check: ensure it is "ready".
	if !framework.WaitForNodeToBeReady(c, name, framework.NodeReadyInitialTimeout) {
		return false
	}

	// Get all the pods on the node that don't have liveness probe set.
	// Liveness probe may cause restart of a pod during node reboot, and the pod may not be running.
	pods := ps.List()
	podNames := []string{}
	for _, p := range pods {
		probe := false
		for _, c := range p.Spec.Containers {
			if c.LivenessProbe != nil {
				probe = true
				break
			}
		}
		if !probe {
			podNames = append(podNames, p.ObjectMeta.Name)
		}
	}
	framework.Logf("Node %s has %d assigned pods with no liveness probes: %v", name, len(podNames), podNames)

	// For each pod, we do a sanity check to ensure it's running / healthy
	// or succeeded now, as that's what we'll be checking later.
	if !framework.CheckPodsRunningReadyOrSucceeded(c, ns, podNames, framework.PodReadyBeforeTimeout) {
		printStatusAndLogsForNotReadyPods(c, ns, podNames, pods)
		return false
	}

	// Reboot the node.
	if err = framework.IssueSSHCommand(rebootCmd, provider, node); err != nil {
		framework.Logf("Error while issuing ssh command: %v", err)
		return false
	}

	// Wait for some kind of "not ready" status.
	if !framework.WaitForNodeToBeNotReady(c, name, rebootNodeNotReadyTimeout) {
		return false
	}

	// Wait for some kind of "ready" status.
	if !framework.WaitForNodeToBeReady(c, name, rebootNodeReadyAgainTimeout) {
		return false
	}

	// Ensure all of the pods that we found on this node before the reboot are
	// running / healthy, or succeeded.
	if !framework.CheckPodsRunningReadyOrSucceeded(c, ns, podNames, rebootPodReadyAgainTimeout) {
		newPods := ps.List()
		printStatusAndLogsForNotReadyPods(c, ns, podNames, newPods)
		return false
	}

	framework.Logf("Reboot successful on node %s", name)
	return true
}

type terminationHook func(provider string, nodes *v1.NodeList)

func catLogHook(logPath string) terminationHook {
	return func(provider string, nodes *v1.NodeList) {
		for _, n := range nodes.Items {
			cmd := fmt.Sprintf("cat %v && rm %v", logPath, logPath)
			if _, err := framework.IssueSSHCommandWithResult(cmd, provider, &n); err != nil {
				framework.Logf("Error while issuing ssh command: %v", err)
			}
		}

	}
}

func dropPacketsScript(chainName, logPath string) string {
	return strings.Replace(fmt.Sprintf(`
		nohup sh -c '
			set -x
			sleep 10
			while true; do sudo iptables -I ${CHAIN} 1 -s 127.0.0.1 -j ACCEPT && break; done
			while true; do sudo iptables -I ${CHAIN} 2 -j DROP && break; done
			date
			sleep 120
			while true; do sudo iptables -D ${CHAIN} -j DROP && break; done
			while true; do sudo iptables -D ${CHAIN} -s 127.0.0.1 -j ACCEPT && break; done
		' >%v 2>&1 &
		`, logPath), "${CHAIN}", chainName, -1)
}
