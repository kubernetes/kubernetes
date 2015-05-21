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
	"os/exec"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/wait"

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

	BeforeEach(func() {
		var err error
		c, err = loadClient()
		Expect(err).NotTo(HaveOccurred())
		ps = newPodStore(c, api.NamespaceDefault, labels.Everything(), fields.Everything())
	})

	AfterEach(func() {
		ps.Stop()
	})

	It("should restart all nodes and ensure all nodes and pods recover", func() {
		// This test requires the ability to restart all nodes, so the provider
		// check must be identical to that call.
		provider := testContext.Provider
		nn := testContext.CloudConfig.NumNodes
		if !providerIs("gce") {
			By(fmt.Sprintf("Skipping reboot test, which is not implemented for %s", provider))
			return
		}

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
		if !checkPodsRunningReady(c, podNamesBefore, podReadyBeforeTimeout) {
			Failf("At least one pod wasn't running and ready at test start.")
		}

		By("restarting all of the nodes")
		err = restartNodes(provider, restartPerNodeTimeout)
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
		if !checkPodsRunningReady(c, podNamesAfter, remaining) {
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
			errLast = fmt.Errorf("expected to find %d nodes but found only %d", expect, len(nodeList.Items))
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
	case "gce":
		return migRollingUpdate(nt)
	default:
		return fmt.Errorf("restartNodes(...) not implemented for %s", provider)
	}
}

// migRollingUpdate starts a MIG rolling update and waits up to nt times the
// nubmer of nodes for it to complete.
func migRollingUpdate(nt time.Duration) error {
	By("getting the name of the template for the managed instance group")
	templ, err := migTemplate()
	if err != nil {
		return fmt.Errorf("couldn't get MIG template name: %v", err)
	}

	By("starting the managed instance group rolling update")
	id, err := migRollingUpdateStart(templ, nt)
	if err != nil {
		return fmt.Errorf("couldn't start the MIG rolling update: %v", err)
	}

	By("polling the managed instance group rolling update until it completes")
	if err := migRollingUpdatePoll(id, nt); err != nil {
		return fmt.Errorf("err waiting until update completed: %v", err)
	}

	return nil
}

// migTemlate (GCE/GKE-only) returns the name of the MIG template that the
// nodes of the cluster use.
func migTemplate() (string, error) {
	var errLast error
	var templ string
	key := "instanceTemplate"
	if wait.Poll(poll, singleCallTimeout, func() (bool, error) {
		// TODO(mbforbes): make this hit the compute API directly instead of
		// shelling out to gcloud.
		o, err := exec.Command("gcloud", "preview", "managed-instance-groups",
			fmt.Sprintf("--project=%s", testContext.CloudConfig.ProjectID),
			fmt.Sprintf("--zone=%s", testContext.CloudConfig.Zone),
			"describe",
			testContext.CloudConfig.NodeInstanceGroup).CombinedOutput()
		if err != nil {
			errLast = fmt.Errorf("gcloud preview managed-instance-groups describe call failed with err: %v", err)
			return false, nil
		}
		output := string(o)

		// The 'describe' call probably succeeded; parse the output and try to
		// find the line that looks like "instanceTemplate: url/to/<templ>" and
		// return <templ>.
		if val := parseKVLines(output, key); len(val) > 0 {
			url := strings.Split(val, "/")
			templ = url[len(url)-1]
			Logf("MIG group %s using template: %s", testContext.CloudConfig.NodeInstanceGroup, templ)
			return true, nil
		}
		errLast = fmt.Errorf("couldn't find %s in output to get MIG template. Output: %s", key, output)
		return false, nil
	}) != nil {
		return "", fmt.Errorf("migTemplate() failed with last error: %v", errLast)
	}
	return templ, nil
}

// migRollingUpdateStart (GCE/GKE-only) starts a MIG rolling update using templ
// as the new template, waiting up to nt per node, and returns the ID of that
// update.
func migRollingUpdateStart(templ string, nt time.Duration) (string, error) {
	var errLast error
	var id string
	prefix, suffix := "Started [", "]."
	if err := wait.Poll(poll, singleCallTimeout, func() (bool, error) {
		// TODO(mbforbes): make this hit the compute API directly instead of
		// shelling out to gcloud.
		o, err := exec.Command("gcloud", "preview", "rolling-updates",
			fmt.Sprintf("--project=%s", testContext.CloudConfig.ProjectID),
			fmt.Sprintf("--zone=%s", testContext.CloudConfig.Zone),
			"start",
			// Required args.
			fmt.Sprintf("--group=%s", testContext.CloudConfig.NodeInstanceGroup),
			fmt.Sprintf("--template=%s", templ),
			// Optional args to fine-tune behavior.
			fmt.Sprintf("--instance-startup-timeout=%ds", int(nt.Seconds())),
			// NOTE: We can speed up this process by increasing
			//       --max-num-concurrent-instances.
			fmt.Sprintf("--max-num-concurrent-instances=%d", 1),
			fmt.Sprintf("--max-num-failed-instances=%d", 0),
			fmt.Sprintf("--min-instance-update-time=%ds", 0)).CombinedOutput()
		if err != nil {
			errLast = fmt.Errorf("gcloud preview rolling-updates call failed with err: %v", err)
			return false, nil
		}
		output := string(o)

		// The 'start' call probably succeeded; parse the output and try to find
		// the line that looks like "Started [url/to/<id>]." and return <id>.
		for _, line := range strings.Split(output, "\n") {
			// As a sanity check, ensure the line starts with prefix and ends
			// with suffix.
			if strings.Index(line, prefix) != 0 || strings.Index(line, suffix) != len(line)-len(suffix) {
				continue
			}
			url := strings.Split(strings.TrimSuffix(strings.TrimPrefix(line, prefix), suffix), "/")
			id = url[len(url)-1]
			Logf("Started MIG rolling update; ID: %s", id)
			return true, nil
		}
		errLast = fmt.Errorf("couldn't find line like '%s ... %s' in output to MIG rolling-update start. Output: %s",
			prefix, suffix, output)
		return false, nil
	}); err != nil {
		return "", fmt.Errorf("migRollingUpdateStart() failed with last error: %v", errLast)
	}
	return id, nil
}

// migRollingUpdatePoll (CKE/GKE-only) polls the progress of the MIG rolling
// update with ID id until it is complete. It returns an error if this takes
// longer than nt times the number of nodes.
func migRollingUpdatePoll(id string, nt time.Duration) error {
	// Two keys and a val.
	status, progress, done := "status", "statusMessage", "ROLLED_OUT"
	start, timeout := time.Now(), nt*time.Duration(testContext.CloudConfig.NumNodes)
	var errLast error
	Logf("Waiting up to %v for MIG rolling update to complete.", timeout)
	if wait.Poll(restartPoll, timeout, func() (bool, error) {
		o, err := exec.Command("gcloud", "preview", "rolling-updates",
			fmt.Sprintf("--project=%s", testContext.CloudConfig.ProjectID),
			fmt.Sprintf("--zone=%s", testContext.CloudConfig.Zone),
			"describe",
			id).CombinedOutput()
		if err != nil {
			errLast = fmt.Errorf("Error calling rolling-updates describe %s: %v", id, err)
			Logf("%v", errLast)
			return false, nil
		}
		output := string(o)

		// The 'describe' call probably succeeded; parse the output and try to
		// find the line that looks like "status: <status>" and see whether it's
		// done.
		Logf("Waiting for MIG rolling update: %s (%v elapsed)",
			parseKVLines(output, progress), time.Since(start))
		if st := parseKVLines(output, status); st == done {
			return true, nil
		}
		return false, nil
	}) != nil {
		return fmt.Errorf("timeout waiting %v for MIG rolling update to complete. Last error: %v", timeout, errLast)
	}
	return nil
}

// parseKVLines parses output that looks like lines containing "<key>: <val>"
// and returns <val> if <key> is found. Otherwise, it returns the empty string.
func parseKVLines(output, key string) string {
	delim := ":"
	key = key + delim
	for _, line := range strings.Split(output, "\n") {
		pieces := strings.SplitAfterN(line, delim, 2)
		if len(pieces) != 2 {
			continue
		}
		k, v := pieces[0], pieces[1]
		if k == key {
			return strings.TrimSpace(v)
		}
	}
	return ""
}
