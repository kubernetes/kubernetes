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
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/wait"
	kubepod "k8s.io/kubernetes/pkg/kubelet/pod"
	"k8s.io/kubernetes/test/e2e/common"
	"k8s.io/kubernetes/test/e2e/framework"
	testutils "k8s.io/kubernetes/test/utils"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

func isNotRestartAlwaysMirrorPod(p *v1.Pod) bool {
	if !kubepod.IsMirrorPod(p) {
		return false
	}
	return p.Spec.RestartPolicy != v1.RestartPolicyAlways
}

func filterIrrelevantPods(pods []*v1.Pod) []*v1.Pod {
	var results []*v1.Pod
	for _, p := range pods {
		if isNotRestartAlwaysMirrorPod(p) {
			// Mirror pods with restart policy == Never will not get
			// recreated if they are deleted after the pods have
			// terminated. For now, we discount such pods.
			// https://github.com/kubernetes/kubernetes/issues/34003
			continue
		}
		results = append(results, p)
	}
	return results
}

func nodeNames(nodes []v1.Node) []string {
	result := make([]string, 0, len(nodes))
	for i := range nodes {
		result = append(result, nodes[i].Name)
	}
	return result
}

var _ = SIGDescribe("Restart [Disruptive]", func() {
	f := framework.NewDefaultFramework("restart")
	var ps *testutils.PodStore
	var originalNodes []v1.Node
	var originalPodNames []string
	var numNodes int
	var systemNamespace string

	BeforeEach(func() {
		// This test requires the ability to restart all nodes, so the provider
		// check must be identical to that call.
		framework.SkipUnlessProviderIs("gce", "gke")
		var err error
		ps, err = testutils.NewPodStore(f.ClientSet, metav1.NamespaceSystem, labels.Everything(), fields.Everything())
		Expect(err).NotTo(HaveOccurred())
		numNodes, err = framework.NumberOfRegisteredNodes(f.ClientSet)
		Expect(err).NotTo(HaveOccurred())
		systemNamespace = metav1.NamespaceSystem

		By("ensuring all nodes are ready")
		originalNodes, err = framework.CheckNodesReady(f.ClientSet, numNodes, framework.NodeReadyInitialTimeout)
		Expect(err).NotTo(HaveOccurred())
		framework.Logf("Got the following nodes before restart: %v", nodeNames(originalNodes))

		By("ensuring all pods are running and ready")
		allPods := ps.List()
		pods := filterIrrelevantPods(allPods)

		originalPodNames = make([]string, len(pods))
		for i, p := range pods {
			originalPodNames[i] = p.ObjectMeta.Name
		}
		if !framework.CheckPodsRunningReadyOrSucceeded(f.ClientSet, systemNamespace, originalPodNames, framework.PodReadyBeforeTimeout) {
			printStatusAndLogsForNotReadyPods(f.ClientSet, systemNamespace, originalPodNames, pods)
			framework.Failf("At least one pod wasn't running and ready or succeeded at test start.")
		}
	})

	AfterEach(func() {
		if ps != nil {
			ps.Stop()
		}
	})

	It("should restart all nodes and ensure all nodes and pods recover", func() {
		By("restarting all of the nodes")
		err := common.RestartNodes(f.ClientSet, originalNodes)
		Expect(err).NotTo(HaveOccurred())

		By("ensuring all nodes are ready after the restart")
		nodesAfter, err := framework.CheckNodesReady(f.ClientSet, numNodes, framework.RestartNodeReadyAgainTimeout)
		Expect(err).NotTo(HaveOccurred())
		framework.Logf("Got the following nodes after restart: %v", nodeNames(nodesAfter))

		// Make sure that we have the same number of nodes. We're not checking
		// that the names match because that's implementation specific.
		By("ensuring the same number of nodes exist after the restart")
		if len(originalNodes) != len(nodesAfter) {
			framework.Failf("Had %d nodes before nodes were restarted, but now only have %d",
				len(originalNodes), len(nodesAfter))
		}

		// Make sure that we have the same number of pods. We're not checking
		// that the names match because they are recreated with different names
		// across node restarts.
		By("ensuring the same number of pods are running and ready after restart")
		podCheckStart := time.Now()
		podNamesAfter, err := waitForNPods(ps, len(originalPodNames), framework.RestartPodReadyAgainTimeout)
		Expect(err).NotTo(HaveOccurred())
		remaining := framework.RestartPodReadyAgainTimeout - time.Since(podCheckStart)
		if !framework.CheckPodsRunningReadyOrSucceeded(f.ClientSet, systemNamespace, podNamesAfter, remaining) {
			pods := ps.List()
			printStatusAndLogsForNotReadyPods(f.ClientSet, systemNamespace, podNamesAfter, pods)
			framework.Failf("At least one pod wasn't running and ready after the restart.")
		}
	})
})

// waitForNPods tries to list pods using c until it finds expect of them,
// returning their names if it can do so before timeout.
func waitForNPods(ps *testutils.PodStore, expect int, timeout time.Duration) ([]string, error) {
	// Loop until we find expect pods or timeout is passed.
	var pods []*v1.Pod
	var errLast error
	found := wait.Poll(framework.Poll, timeout, func() (bool, error) {
		allPods := ps.List()
		pods = filterIrrelevantPods(allPods)
		if len(pods) != expect {
			errLast = fmt.Errorf("expected to find %d pods but found only %d", expect, len(pods))
			framework.Logf("Error getting pods: %v", errLast)
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
