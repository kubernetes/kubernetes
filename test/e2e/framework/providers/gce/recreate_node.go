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

package gce

import (
	"fmt"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	testutils "k8s.io/kubernetes/test/utils"
)

func nodeNames(nodes []v1.Node) []string {
	result := make([]string, 0, len(nodes))
	for i := range nodes {
		result = append(result, nodes[i].Name)
	}
	return result
}

func podNames(pods []v1.Pod) []string {
	result := make([]string, 0, len(pods))
	for i := range pods {
		result = append(result, pods[i].Name)
	}
	return result
}

var _ = Describe("Recreate [Feature:Recreate]", func() {
	f := framework.NewDefaultFramework("recreate")
	var originalNodes []v1.Node
	var originalPodNames []string
	var ps *testutils.PodStore
	systemNamespace := metav1.NamespaceSystem
	BeforeEach(func() {
		framework.SkipUnlessProviderIs("gce", "gke")
		var err error
		numNodes, err := framework.NumberOfRegisteredNodes(f.ClientSet)
		Expect(err).NotTo(HaveOccurred())
		originalNodes, err = framework.CheckNodesReady(f.ClientSet, numNodes, framework.NodeReadyInitialTimeout)
		Expect(err).NotTo(HaveOccurred())

		framework.Logf("Got the following nodes before recreate %v", nodeNames(originalNodes))

		ps, err = testutils.NewPodStore(f.ClientSet, systemNamespace, labels.Everything(), fields.Everything())
		allPods := ps.List()
		originalPods := framework.FilterNonRestartablePods(allPods)
		originalPodNames = make([]string, len(originalPods))
		for i, p := range originalPods {
			originalPodNames[i] = p.ObjectMeta.Name
		}

		if !framework.CheckPodsRunningReadyOrSucceeded(f.ClientSet, systemNamespace, originalPodNames, framework.PodReadyBeforeTimeout) {
			framework.Failf("At least one pod wasn't running and ready or succeeded at test start.")
		}

	})

	AfterEach(func() {
		if CurrentGinkgoTestDescription().Failed {
			// Make sure that addon/system pods are running, so dump
			// events for the kube-system namespace on failures
			By(fmt.Sprintf("Collecting events from namespace %q.", systemNamespace))
			events, err := f.ClientSet.CoreV1().Events(systemNamespace).List(metav1.ListOptions{})
			Expect(err).NotTo(HaveOccurred())

			for _, e := range events.Items {
				framework.Logf("event for %v: %v %v: %v", e.InvolvedObject.Name, e.Source, e.Reason, e.Message)
			}
		}
		if ps != nil {
			ps.Stop()
		}
	})

	It("recreate nodes and ensure they function upon restart", func() {
		testRecreate(f.ClientSet, ps, systemNamespace, originalNodes, originalPodNames)
	})
})

// Recreate all the nodes in the test instance group
func testRecreate(c clientset.Interface, ps *testutils.PodStore, systemNamespace string, nodes []v1.Node, podNames []string) {
	err := recreateNodes(c, nodes)
	if err != nil {
		framework.Failf("Test failed; failed to start the restart instance group command.")
	}

	err = waitForNodeBootIdsToChange(c, nodes, framework.RecreateNodeReadyAgainTimeout)
	if err != nil {
		framework.Failf("Test failed; failed to recreate at least one node in %v.", framework.RecreateNodeReadyAgainTimeout)
	}

	nodesAfter, err := framework.CheckNodesReady(c, len(nodes), framework.RestartNodeReadyAgainTimeout)
	Expect(err).NotTo(HaveOccurred())
	framework.Logf("Got the following nodes after recreate: %v", nodeNames(nodesAfter))

	if len(nodes) != len(nodesAfter) {
		framework.Failf("Had %d nodes before nodes were recreated, but now only have %d",
			len(nodes), len(nodesAfter))
	}

	// Make sure the pods from before node recreation are running/completed
	podCheckStart := time.Now()
	podNamesAfter, err := framework.WaitForNRestartablePods(ps, len(podNames), framework.RestartPodReadyAgainTimeout)
	Expect(err).NotTo(HaveOccurred())
	remaining := framework.RestartPodReadyAgainTimeout - time.Since(podCheckStart)
	if !framework.CheckPodsRunningReadyOrSucceeded(c, systemNamespace, podNamesAfter, remaining) {
		framework.Failf("At least one pod wasn't running and ready after the restart.")
	}
}
