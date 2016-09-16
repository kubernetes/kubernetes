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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("Restart [Disruptive]", func() {
	f := framework.NewDefaultFramework("restart")
	var ps *framework.PodStore

	BeforeEach(func() {
		// This test requires the ability to restart all nodes, so the provider
		// check must be identical to that call.
		framework.SkipUnlessProviderIs("gce", "gke")

		ps = framework.NewPodStore(f.Client, api.NamespaceSystem, labels.Everything(), fields.Everything())
	})

	AfterEach(func() {
		if ps != nil {
			ps.Stop()
		}
	})

	It("should restart all nodes and ensure all nodes and pods recover", func() {
		nn := framework.TestContext.CloudConfig.NumNodes

		By("ensuring all nodes are ready")
		nodeNamesBefore, err := framework.CheckNodesReady(f.Client, framework.NodeReadyInitialTimeout, nn)
		Expect(err).NotTo(HaveOccurred())
		framework.Logf("Got the following nodes before restart: %v", nodeNamesBefore)

		By("ensuring all pods are running and ready")
		pods := ps.List()
		podNamesBefore := make([]string, len(pods))
		for i, p := range pods {
			podNamesBefore[i] = p.ObjectMeta.Name
		}
		ns := api.NamespaceSystem
		if !framework.CheckPodsRunningReadyOrSucceeded(f.Client, ns, podNamesBefore, framework.PodReadyBeforeTimeout) {
			framework.Failf("At least one pod wasn't running and ready or succeeded at test start.")
		}

		By("restarting all of the nodes")
		err = restartNodes(framework.TestContext.Provider, framework.RestartPerNodeTimeout)
		Expect(err).NotTo(HaveOccurred())

		By("ensuring all nodes are ready after the restart")
		nodeNamesAfter, err := framework.CheckNodesReady(f.Client, framework.RestartNodeReadyAgainTimeout, nn)
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
		if !framework.CheckPodsRunningReadyOrSucceeded(f.Client, ns, podNamesAfter, remaining) {
			framework.Failf("At least one pod wasn't running and ready after the restart.")
		}
	})
})

// waitForNPods tries to list pods using c until it finds expect of them,
// returning their names if it can do so before timeout.
func waitForNPods(ps *framework.PodStore, expect int, timeout time.Duration) ([]string, error) {
	// Loop until we find expect pods or timeout is passed.
	var pods []*api.Pod
	var errLast error
	found := wait.Poll(framework.Poll, timeout, func() (bool, error) {
		pods = ps.List()
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
	tmpl, err := framework.MigTemplate()
	if err != nil {
		return fmt.Errorf("couldn't get MIG template name: %v", err)
	}
	return framework.MigRollingUpdate(tmpl, nt)
}
