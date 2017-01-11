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
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/fields"
	kubepod "k8s.io/kubernetes/pkg/kubelet/pod"
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

var _ = framework.KubeDescribe("Restart [Disruptive]", func() {
	f := framework.NewDefaultFramework("restart")
	var ps *testutils.PodStore

	BeforeEach(func() {
		// This test requires the ability to restart all nodes, so the provider
		// check must be identical to that call.
		framework.SkipUnlessProviderIs("gce", "gke")

		ps = testutils.NewPodStore(f.ClientSet, api.NamespaceSystem, labels.Everything(), fields.Everything())
	})

	AfterEach(func() {
		if ps != nil {
			ps.Stop()
		}
	})

	It("should restart all nodes and ensure all nodes and pods recover", func() {
		nn := framework.TestContext.CloudConfig.NumNodes

		By("ensuring all nodes are ready")
		nodeNamesBefore, err := framework.CheckNodesReady(f.ClientSet, framework.NodeReadyInitialTimeout, nn)
		Expect(err).NotTo(HaveOccurred())
		framework.Logf("Got the following nodes before restart: %v", nodeNamesBefore)

		By("ensuring all pods are running and ready")
		allPods := ps.List()
		pods := filterIrrelevantPods(allPods)

		podNamesBefore := make([]string, len(pods))
		for i, p := range pods {
			podNamesBefore[i] = p.ObjectMeta.Name
		}
		ns := api.NamespaceSystem
		if !framework.CheckPodsRunningReadyOrSucceeded(f.ClientSet, ns, podNamesBefore, framework.PodReadyBeforeTimeout) {
			framework.Failf("At least one pod wasn't running and ready or succeeded at test start.")
		}

		By("restarting all of the nodes")
		err = restartNodes(f, nodeNamesBefore)
		Expect(err).NotTo(HaveOccurred())

		By("ensuring all nodes are ready after the restart")
		nodeNamesAfter, err := framework.CheckNodesReady(f.ClientSet, framework.RestartNodeReadyAgainTimeout, nn)
		Expect(err).NotTo(HaveOccurred())
		framework.Logf("Got the following nodes after restart: %v", nodeNamesAfter)

		// Make sure that we have the same number of nodes. We're not checking
		// that the names match because that's implementation specific.
		By("ensuring the same number of nodes exist after the restart")
		if len(nodeNamesBefore) != len(nodeNamesAfter) {
			framework.Failf("Had %d nodes before nodes were restarted, but now only have %d",
				len(nodeNamesBefore), len(nodeNamesAfter))
		}

		// Make sure that we have the same number of pods. We're not checking
		// that the names match because they are recreated with different names
		// across node restarts.
		By("ensuring the same number of pods are running and ready after restart")
		podCheckStart := time.Now()
		podNamesAfter, err := waitForNPods(ps, len(podNamesBefore), framework.RestartPodReadyAgainTimeout)
		Expect(err).NotTo(HaveOccurred())
		remaining := framework.RestartPodReadyAgainTimeout - time.Since(podCheckStart)
		if !framework.CheckPodsRunningReadyOrSucceeded(f.ClientSet, ns, podNamesAfter, remaining) {
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
		pods := filterIrrelevantPods(allPods)
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

func restartNodes(f *framework.Framework, nodeNames []string) error {
	// List old boot IDs.
	oldBootIDs := make(map[string]string)
	for _, name := range nodeNames {
		node, err := f.ClientSet.Core().Nodes().Get(name, metav1.GetOptions{})
		if err != nil {
			return fmt.Errorf("error getting node info before reboot: %s", err)
		}
		oldBootIDs[name] = node.Status.NodeInfo.BootID
	}
	// Reboot the nodes.
	args := []string{
		"compute",
		fmt.Sprintf("--project=%s", framework.TestContext.CloudConfig.ProjectID),
		"instances",
		"reset",
	}
	args = append(args, nodeNames...)
	args = append(args, fmt.Sprintf("--zone=%s", framework.TestContext.CloudConfig.Zone))
	stdout, stderr, err := framework.RunCmd("gcloud", args...)
	if err != nil {
		return fmt.Errorf("error restarting nodes: %s\nstdout: %s\nstderr: %s", err, stdout, stderr)
	}
	// Wait for their boot IDs to change.
	for _, name := range nodeNames {
		if err := wait.Poll(30*time.Second, 5*time.Minute, func() (bool, error) {
			node, err := f.ClientSet.Core().Nodes().Get(name, metav1.GetOptions{})
			if err != nil {
				return false, fmt.Errorf("error getting node info after reboot: %s", err)
			}
			return node.Status.NodeInfo.BootID != oldBootIDs[name], nil
		}); err != nil {
			return fmt.Errorf("error waiting for node %s boot ID to change: %s", name, err)
		}
	}
	return nil
}
