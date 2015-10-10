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

	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util/wait"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	// How long each node is given during a process that restarts all nodes
	// before the test is considered failed. (Note that the total time to
	// restart all nodes will be this number times the nubmer of nodes.)
	restartPerNodeTimeout = 5 * time.Minute

	// How often to poll the statues of a restart.
	restartPoll = 20 * time.Second

	// How long a node is allowed to become "Ready" after it is restarted before
	// the test is considered failed.
	restartNodeReadyAgainTimeout = 5 * time.Minute

	// How long a pod is allowed to become "running" and "ready" after a node
	// restart before test is considered failed.
	restartPodReadyAgainTimeout = 5 * time.Minute
)

var _ = Describe("Restart", func() {
	var c *client.Client
	var ps *podStore
	var skipped bool

	BeforeEach(func() {
		var err error
		c, err = loadClient()
		Expect(err).NotTo(HaveOccurred())

		// This test requires the ability to restart all nodes, so the provider
		// check must be identical to that call.
		skipped = true
		SkipUnlessProviderIs("gce", "gke")
		skipped = false

		ps = newPodStore(c, api.NamespaceSystem, labels.Everything(), fields.Everything())
	})

	AfterEach(func() {
		if skipped {
			return
		}

		ps.Stop()
	})

	It("should restart all nodes and ensure all nodes and pods recover", func() {
		nn := testContext.CloudConfig.NumNodes

		By("ensuring all nodes are ready")
		nodeNamesBefore, err := checkNodesReady(c, nodeReadyInitialTimeout, nn)
		Expect(err).NotTo(HaveOccurred())
		Logf("Got the following nodes before restart: %v", nodeNamesBefore)

		By("ensuring all pods are running and ready")
		pods := ps.List()
		podNamesBefore := make([]string, len(pods))
		for i, p := range pods {
			podNamesBefore[i] = p.ObjectMeta.Name
		}
		ns := api.NamespaceSystem
		if !checkPodsRunningReady(c, ns, podNamesBefore, podReadyBeforeTimeout) {
			Failf("At least one pod wasn't running and ready at test start.")
		}

		By("restarting all of the nodes")
		err = restartNodes(testContext.Provider, restartPerNodeTimeout)
		Expect(err).NotTo(HaveOccurred())

		By("ensuring all nodes are ready after the restart")
		nodeNamesAfter, err := checkNodesReady(c, restartNodeReadyAgainTimeout, nn)
		Expect(err).NotTo(HaveOccurred())
		Logf("Got the following nodes after restart: %v", nodeNamesAfter)

		// Make sure that we have the same number of nodes. We're not checking
		// that the names match because that's implementation specific.
		By("ensuring the same number of nodes exist after the restart")
		if len(nodeNamesBefore) != len(nodeNamesAfter) {
			Failf("Had %d nodes before nodes were restarted, but now only have %d",
				len(nodeNamesBefore), len(nodeNamesAfter))
		}

		// Make sure that we have the same number of pods. We're not checking
		// that the names match because they are recreated with different names
		// across node restarts.
		By("ensuring the same number of pods are running and ready after restart")
		podCheckStart := time.Now()
		podNamesAfter, err := waitForNPods(ps, len(podNamesBefore), restartPodReadyAgainTimeout)
		Expect(err).NotTo(HaveOccurred())
		remaining := restartPodReadyAgainTimeout - time.Since(podCheckStart)
		if !checkPodsRunningReady(c, ns, podNamesAfter, remaining) {
			Failf("At least one pod wasn't running and ready after the restart.")
		}
	})
})

// waitForNPods tries to list pods using c until it finds expect of them,
// returning their names if it can do so before timeout.
func waitForNPods(ps *podStore, expect int, timeout time.Duration) ([]string, error) {
	// Loop until we find expect pods or timeout is passed.
	var pods []*api.Pod
	var errLast error
	found := wait.Poll(poll, timeout, func() (bool, error) {
		pods = ps.List()
		if len(pods) != expect {
			errLast = fmt.Errorf("expected to find %d pods but found only %d", expect, len(pods))
			Logf("Error getting pods: %v", errLast)
			return false, nil
		}
		return true, nil
	}) == nil
	// Extract the names of all found pods.
	podNames := make([]string, len(pods))
	for i, p := range pods {
		podNames[i] = p.ObjectMeta.Name
	}
	if !found {
		return podNames, fmt.Errorf("couldn't find %d pods within %v; last error: %v",
			expect, timeout, errLast)
	}
	return podNames, nil
}

// checkNodesReady waits up to nt for expect nodes accessed by c to be ready,
// returning an error if this doesn't happen in time. It returns the names of
// nodes it finds.
func checkNodesReady(c *client.Client, nt time.Duration, expect int) ([]string, error) {
	// First, keep getting all of the nodes until we get the number we expect.
	var nodeList *api.NodeList
	var errLast error
	start := time.Now()
	found := wait.Poll(poll, nt, func() (bool, error) {
		// Even though listNodes(...) has its own retries, a rolling-update
		// (GCE/GKE implementation of restart) can complete before the apiserver
		// knows about all of the nodes. Thus, we retry the list nodes call
		// until we get the expected number of nodes.
		nodeList, errLast = listNodes(c, labels.Everything(), fields.Everything())
		if errLast != nil {
			return false, nil
		}
		if len(nodeList.Items) != expect {
			errLast = fmt.Errorf("expected to find %d nodes but found only %d (%v elapsed)",
				expect, len(nodeList.Items), time.Since(start))
			Logf("%v", errLast)
			return false, nil
		}
		return true, nil
	}) == nil
	nodeNames := make([]string, len(nodeList.Items))
	for i, n := range nodeList.Items {
		nodeNames[i] = n.ObjectMeta.Name
	}
	if !found {
		return nodeNames, fmt.Errorf("couldn't find %d nodes within %v; last error: %v",
			expect, nt, errLast)
	}
	Logf("Successfully found %d nodes", expect)

	// Next, ensure in parallel that all the nodes are ready. We subtract the
	// time we spent waiting above.
	timeout := nt - time.Since(start)
	result := make(chan bool, len(nodeList.Items))
	for _, n := range nodeNames {
		n := n
		go func() { result <- waitForNodeToBeReady(c, n, timeout) }()
	}
	failed := false
	// TODO(mbforbes): Change to `for range` syntax once we support only Go
	// >= 1.4.
	for i := range nodeList.Items {
		_ = i
		if !<-result {
			failed = true
		}
	}
	if failed {
		return nodeNames, fmt.Errorf("at least one node failed to be ready")
	}
	return nodeNames, nil
}

// restartNodes uses provider to do a restart of all nodes in the cluster,
// allowing up to nt per node.
func restartNodes(provider string, nt time.Duration) error {
	switch provider {
	case "gce", "gke":
		return migRollingUpdateSelf(nt)
	default:
		return fmt.Errorf("restartNodes(...) not implemented for %s", provider)
	}
}

// TODO(marekbiskup): Switch this to MIG recreate-instances. This can be done
// with the following bash, but needs to be written in Go:
//
//   # Step 1: Get instance names.
//   list=$(gcloud compute instance-groups --project=${PROJECT} --zone=${ZONE} instances --group=${GROUP} list)
//   i=""
//   for l in $list; do
// 	  i="${l##*/},${i}"
//   done
//
//   # Step 2: Start the recreate.
//   output=$(gcloud compute instance-groups managed --project=${PROJECT} --zone=${ZONE} recreate-instances ${GROUP} --instance="${i}")
//   op=${output##*:}
//
//   # Step 3: Wait until it's complete.
//   status=""
//   while [[ "${status}" != "DONE" ]]; do
// 	  output=$(gcloud compute instance-groups managed --zone="${ZONE}" get-operation ${op} | grep status)
// 	  status=${output##*:}
//   done
func migRollingUpdateSelf(nt time.Duration) error {
	By("getting the name of the template for the managed instance group")
	tmpl, err := migTemplate()
	if err != nil {
		return fmt.Errorf("couldn't get MIG template name: %v", err)
	}
	return migRollingUpdate(tmpl, nt)
}
