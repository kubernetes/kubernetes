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
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"

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

var _ = Describe("Reboot", func() {
	var f *Framework

	BeforeEach(func() {
		// These tests requires SSH to nodes, so the provider check should be identical to there
		// (the limiting factor is the implementation of util.go's getSigner(...)).

		// Cluster must support node reboot
		SkipUnlessProviderIs(providersWithSSH...)
	})

	AfterEach(func() {
		if CurrentGinkgoTestDescription().Failed {
			// Most of the reboot tests just make sure that addon/system pods are running, so dump
			// events for the kube-system namespace on failures
			namespaceName := api.NamespaceSystem
			By(fmt.Sprintf("Collecting events from namespace %q.", namespaceName))
			events, err := f.Client.Events(namespaceName).List(labels.Everything(), fields.Everything())
			Expect(err).NotTo(HaveOccurred())

			for _, e := range events.Items {
				Logf("event for %v: %v %v: %v", e.InvolvedObject.Name, e.Source, e.Reason, e.Message)
			}
		}
	})

	f = NewFramework("reboot")

	It("each node by ordering clean reboot and ensure they function upon restart", func() {
		// clean shutdown and restart
		// We sleep 10 seconds to give some time for ssh command to cleanly finish before the node is rebooted.
		testReboot(f.Client, "nohup sh -c 'sleep 10 && sudo reboot' >/dev/null 2>&1 &")
	})

	It("each node by ordering unclean reboot and ensure they function upon restart", func() {
		// unclean shutdown and restart
		// We sleep 10 seconds to give some time for ssh command to cleanly finish before the node is shutdown.
		testReboot(f.Client, "nohup sh -c 'sleep 10 && echo b | sudo tee /proc/sysrq-trigger' >/dev/null 2>&1 &")
	})

	It("each node by triggering kernel panic and ensure they function upon restart", func() {
		// kernel panic
		// We sleep 10 seconds to give some time for ssh command to cleanly finish before kernel panic is triggered.
		testReboot(f.Client, "nohup sh -c 'sleep 10 && echo c | sudo tee /proc/sysrq-trigger' >/dev/null 2>&1 &")
	})

	It("each node by switching off the network interface and ensure they function upon switch on", func() {
		// switch the network interface off for a while to simulate a network outage
		// We sleep 10 seconds to give some time for ssh command to cleanly finish before network is down.
		testReboot(f.Client, "nohup sh -c 'sleep 10 && sudo ifdown eth0 && sleep 120 && sudo ifup eth0' >/dev/null 2>&1 &")
	})

	It("each node by dropping all inbound packets for a while and ensure they function afterwards", func() {
		// tell the firewall to drop all inbound packets for a while
		// We sleep 10 seconds to give some time for ssh command to cleanly finish before starting dropping inbound packets.
		// We still accept packages send from localhost to prevent monit from restarting kubelet.
		testReboot(f.Client, "nohup sh -c 'sleep 10 && sudo iptables -I INPUT 1 -s 127.0.0.1 -j ACCEPT && sudo iptables -I INPUT 2 -j DROP && "+
			" sleep 120 && sudo iptables -D INPUT -j DROP && sudo iptables -D INPUT -s 127.0.0.1 -j ACCEPT' >/dev/null 2>&1 &")
	})

	It("each node by dropping all outbound packets for a while and ensure they function afterwards", func() {
		// tell the firewall to drop all outbound packets for a while
		// We sleep 10 seconds to give some time for ssh command to cleanly finish before starting dropping outbound packets.
		// We still accept packages send to localhost to prevent monit from restarting kubelet.
		testReboot(f.Client, "nohup sh -c 'sleep 10 &&  sudo iptables -I OUTPUT 1 -s 127.0.0.1 -j ACCEPT && sudo iptables -I OUTPUT 2 -j DROP && "+
			" sleep 120 && sudo iptables -D OUTPUT -j DROP && sudo iptables -D OUTPUT -s 127.0.0.1 -j ACCEPT' >/dev/null 2>&1 &")
	})
})

func testReboot(c *client.Client, rebootCmd string) {
	// Get all nodes, and kick off the test on each.
	nodelist, err := listNodes(c, labels.Everything(), fields.Everything())
	if err != nil {
		Failf("Error getting nodes: %v", err)
	}
	result := make([]bool, len(nodelist.Items))
	wg := sync.WaitGroup{}
	wg.Add(len(nodelist.Items))

	failed := false
	for ix := range nodelist.Items {
		go func(ix int) {
			defer wg.Done()
			n := nodelist.Items[ix]
			result[ix] = rebootNode(c, testContext.Provider, n.ObjectMeta.Name, rebootCmd)
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
				Logf("Node %s failed reboot test.", n.ObjectMeta.Name)
			}
		}
		Failf("Test failed; at least one node failed to reboot in the time given.")
	}
}

func issueSSHCommand(node *api.Node, provider, cmd string) error {
	Logf("Getting external IP address for %s", node.Name)
	host := ""
	for _, a := range node.Status.Addresses {
		if a.Type == api.NodeExternalIP {
			host = a.Address + ":22"
			break
		}
	}
	if host == "" {
		return fmt.Errorf("couldn't find external IP address for node %s", node.Name)
	}
	Logf("Calling %s on %s", cmd, node.Name)
	if _, _, code, err := SSH(cmd, host, provider); code != 0 || err != nil {
		return fmt.Errorf("when running %s on %s, got %d and %v", cmd, node.Name, code, err)
	}
	return nil
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
func rebootNode(c *client.Client, provider, name, rebootCmd string) bool {
	// Setup
	ns := api.NamespaceSystem
	ps := newPodStore(c, ns, labels.Everything(), fields.OneTermEqualSelector(client.PodHost, name))
	defer ps.Stop()

	// Get the node initially.
	Logf("Getting %s", name)
	node, err := c.Nodes().Get(name)
	if err != nil {
		Logf("Couldn't get node %s", name)
		return false
	}

	// Node sanity check: ensure it is "ready".
	if !waitForNodeToBeReady(c, name, nodeReadyInitialTimeout) {
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
	Logf("Node %s has %d pods: %v", name, len(podNames), podNames)

	// For each pod, we do a sanity check to ensure it's running / healthy
	// now, as that's what we'll be checking later.
	if !checkPodsRunningReady(c, ns, podNames, podReadyBeforeTimeout) {
		return false
	}

	// Reboot the node.
	if err = issueSSHCommand(node, provider, rebootCmd); err != nil {
		Logf("Error while issuing ssh command: %v", err)
		return false
	}

	// Wait for some kind of "not ready" status.
	if !waitForNodeToBeNotReady(c, name, rebootNodeNotReadyTimeout) {
		return false
	}

	// Wait for some kind of "ready" status.
	if !waitForNodeToBeReady(c, name, rebootNodeReadyAgainTimeout) {
		return false
	}

	// Ensure all of the pods that we found on this node before the reboot are
	// running / healthy.
	if !checkPodsRunningReady(c, ns, podNames, rebootPodReadyAgainTimeout) {
		return false
	}

	Logf("Reboot successful on node %s", name)
	return true
}
