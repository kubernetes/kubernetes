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

package e2e

import (
	"fmt"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
	testutils "k8s.io/kubernetes/test/utils"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	// Interval to framework.Poll /runningpods on a node
	pollInterval = 1 * time.Second
	// Interval to framework.Poll /stats/container on a node
	containerStatsPollingInterval = 5 * time.Second
	// Maximum number of nodes that we constraint to
	maxNodesToCheck = 10
)

// getPodMatches returns a set of pod names on the given node that matches the
// podNamePrefix and namespace.
func getPodMatches(c clientset.Interface, nodeName string, podNamePrefix string, namespace string) sets.String {
	matches := sets.NewString()
	framework.Logf("Checking pods on node %v via /runningpods endpoint", nodeName)
	runningPods, err := framework.GetKubeletPods(c, nodeName)
	if err != nil {
		framework.Logf("Error checking running pods on %v: %v", nodeName, err)
		return matches
	}
	for _, pod := range runningPods.Items {
		if pod.Namespace == namespace && strings.HasPrefix(pod.Name, podNamePrefix) {
			matches.Insert(pod.Name)
		}
	}
	return matches
}

// waitTillNPodsRunningOnNodes polls the /runningpods endpoint on kubelet until
// it finds targetNumPods pods that match the given criteria (namespace and
// podNamePrefix). Note that we usually use label selector to filter pods that
// belong to the same RC. However, we use podNamePrefix with namespace here
// because pods returned from /runningpods do not contain the original label
// information; they are reconstructed by examining the container runtime. In
// the scope of this test, we do not expect pod naming conflicts so
// podNamePrefix should be sufficient to identify the pods.
func waitTillNPodsRunningOnNodes(c clientset.Interface, nodeNames sets.String, podNamePrefix string, namespace string, targetNumPods int, timeout time.Duration) error {
	return wait.Poll(pollInterval, timeout, func() (bool, error) {
		matchCh := make(chan sets.String, len(nodeNames))
		for _, item := range nodeNames.List() {
			// Launch a goroutine per node to check the pods running on the nodes.
			nodeName := item
			go func() {
				matchCh <- getPodMatches(c, nodeName, podNamePrefix, namespace)
			}()
		}

		seen := sets.NewString()
		for i := 0; i < len(nodeNames.List()); i++ {
			seen = seen.Union(<-matchCh)
		}
		if seen.Len() == targetNumPods {
			return true, nil
		}
		framework.Logf("Waiting for %d pods to be running on the node; %d are currently running;", targetNumPods, seen.Len())
		return false, nil
	})
}

// updates labels of nodes given by nodeNames.
// In case a given label already exists, it overwrites it. If label to remove doesn't exist
// it silently ignores it.
// TODO: migrate to use framework.AddOrUpdateLabelOnNode/framework.RemoveLabelOffNode
func updateNodeLabels(c clientset.Interface, nodeNames sets.String, toAdd, toRemove map[string]string) {
	const maxRetries = 5
	for nodeName := range nodeNames {
		var node *api.Node
		var err error
		for i := 0; i < maxRetries; i++ {
			node, err = c.Core().Nodes().Get(nodeName)
			if err != nil {
				framework.Logf("Error getting node %s: %v", nodeName, err)
				continue
			}
			if toAdd != nil {
				for k, v := range toAdd {
					node.ObjectMeta.Labels[k] = v
				}
			}
			if toRemove != nil {
				for k := range toRemove {
					delete(node.ObjectMeta.Labels, k)
				}
			}
			_, err = c.Core().Nodes().Update(node)
			if err != nil {
				framework.Logf("Error updating node %s: %v", nodeName, err)
			} else {
				break
			}
		}
		Expect(err).NotTo(HaveOccurred())
	}
}

var _ = framework.KubeDescribe("kubelet", func() {
	var c clientset.Interface
	var numNodes int
	var nodeNames sets.String
	var nodeLabels map[string]string
	f := framework.NewDefaultFramework("kubelet")
	var resourceMonitor *framework.ResourceMonitor

	BeforeEach(func() {
		c = f.ClientSet
		// Use node labels to restrict the pods to be assigned only to the
		// nodes we observe initially.
		nodeLabels = make(map[string]string)
		nodeLabels["kubelet_cleanup"] = "true"

		nodes := framework.GetReadySchedulableNodesOrDie(c)
		numNodes = len(nodes.Items)
		nodeNames = sets.NewString()
		// If there are a lot of nodes, we don't want to use all of them
		// (if there are 1000 nodes in the cluster, starting 10 pods/node
		// will take ~10 minutes today). And there is also deletion phase.
		// Instead, we choose at most 10 nodes.
		if numNodes > maxNodesToCheck {
			numNodes = maxNodesToCheck
		}
		for i := 0; i < numNodes; i++ {
			nodeNames.Insert(nodes.Items[i].Name)
		}
		updateNodeLabels(c, nodeNames, nodeLabels, nil)

		// Start resourceMonitor only in small clusters.
		if len(nodes.Items) <= maxNodesToCheck {
			resourceMonitor = framework.NewResourceMonitor(f.ClientSet, framework.TargetContainers(), containerStatsPollingInterval)
			resourceMonitor.Start()
		}
	})

	AfterEach(func() {
		if resourceMonitor != nil {
			resourceMonitor.Stop()
		}
		// If we added labels to nodes in this test, remove them now.
		updateNodeLabels(c, nodeNames, nil, nodeLabels)
	})

	framework.KubeDescribe("Clean up pods on node", func() {
		type DeleteTest struct {
			podsPerNode int
			timeout     time.Duration
		}
		deleteTests := []DeleteTest{
			{podsPerNode: 10, timeout: 1 * time.Minute},
		}
		for _, itArg := range deleteTests {
			name := fmt.Sprintf(
				"kubelet should be able to delete %d pods per node in %v.", itArg.podsPerNode, itArg.timeout)
			It(name, func() {
				totalPods := itArg.podsPerNode * numNodes
				By(fmt.Sprintf("Creating a RC of %d pods and wait until all pods of this RC are running", totalPods))
				rcName := fmt.Sprintf("cleanup%d-%s", totalPods, string(uuid.NewUUID()))

				Expect(framework.RunRC(testutils.RCConfig{
					Client:       f.ClientSet,
					Name:         rcName,
					Namespace:    f.Namespace.Name,
					Image:        framework.GetPauseImageName(f.ClientSet),
					Replicas:     totalPods,
					NodeSelector: nodeLabels,
				})).NotTo(HaveOccurred())
				// Perform a sanity check so that we know all desired pods are
				// running on the nodes according to kubelet. The timeout is set to
				// only 30 seconds here because framework.RunRC already waited for all pods to
				// transition to the running status.
				Expect(waitTillNPodsRunningOnNodes(f.ClientSet, nodeNames, rcName, f.Namespace.Name, totalPods,
					time.Second*30)).NotTo(HaveOccurred())
				if resourceMonitor != nil {
					resourceMonitor.LogLatest()
				}

				By("Deleting the RC")
				framework.DeleteRCAndPods(f.ClientSet, f.Namespace.Name, rcName)
				// Check that the pods really are gone by querying /runningpods on the
				// node. The /runningpods handler checks the container runtime (or its
				// cache) and  returns a list of running pods. Some possible causes of
				// failures are:
				//   - kubelet deadlock
				//   - a bug in graceful termination (if it is enabled)
				//   - docker slow to delete pods (or resource problems causing slowness)
				start := time.Now()
				Expect(waitTillNPodsRunningOnNodes(f.ClientSet, nodeNames, rcName, f.Namespace.Name, 0,
					itArg.timeout)).NotTo(HaveOccurred())
				framework.Logf("Deleting %d pods on %d nodes completed in %v after the RC was deleted", totalPods, len(nodeNames),
					time.Since(start))
				if resourceMonitor != nil {
					resourceMonitor.LogCPUSummary()
				}
			})
		}
	})
})
