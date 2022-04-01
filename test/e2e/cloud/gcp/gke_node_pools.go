/*
Copyright 2016 The Kubernetes Authors.

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

package gcp

import (
	"fmt"
	e2econfig "k8s.io/kubernetes/test/e2e/framework/config"
	e2eutils "k8s.io/kubernetes/test/e2e/framework/utils"
	"os/exec"

	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"

	"github.com/onsi/ginkgo"
)

var _ = SIGDescribe("GKE node pools [Feature:GKENodePool]", func() {

	f := framework.NewDefaultFramework("node-pools")

	ginkgo.BeforeEach(func() {
		e2eskipper.SkipUnlessProviderIs("gke")
	})

	ginkgo.It("should create a cluster with multiple node pools [Feature:GKENodePool]", func() {
		e2eutils.Logf("Start create node pool test")
		testCreateDeleteNodePool(f, "test-pool")
	})
})

func testCreateDeleteNodePool(f *framework.Framework, poolName string) {
	e2eutils.Logf("Create node pool: %q in cluster: %q", poolName, e2econfig.TestContext.CloudConfig.Cluster)

	clusterStr := fmt.Sprintf("--cluster=%s", e2econfig.TestContext.CloudConfig.Cluster)

	out, err := exec.Command("gcloud", "container", "node-pools", "create",
		poolName,
		clusterStr,
		"--num-nodes=2").CombinedOutput()
	e2eutils.Logf("\n%s", string(out))
	if err != nil {
		e2eutils.Failf("Failed to create node pool %q. Err: %v\n%v", poolName, err, string(out))
	}
	e2eutils.Logf("Successfully created node pool %q.", poolName)

	out, err = exec.Command("gcloud", "container", "node-pools", "list",
		clusterStr).CombinedOutput()
	if err != nil {
		e2eutils.Failf("Failed to list node pools from cluster %q. Err: %v\n%v", e2econfig.TestContext.CloudConfig.Cluster, err, string(out))
	}
	e2eutils.Logf("Node pools:\n%s", string(out))

	e2eutils.Logf("Checking that 2 nodes have the correct node pool label.")
	nodeCount := nodesWithPoolLabel(f, poolName)
	if nodeCount != 2 {
		e2eutils.Failf("Wanted 2 nodes with node pool label, got: %v", nodeCount)
	}
	e2eutils.Logf("Success, found 2 nodes with correct node pool labels.")

	e2eutils.Logf("Deleting node pool: %q in cluster: %q", poolName, e2econfig.TestContext.CloudConfig.Cluster)
	out, err = exec.Command("gcloud", "container", "node-pools", "delete",
		poolName,
		clusterStr,
		"-q").CombinedOutput()
	e2eutils.Logf("\n%s", string(out))
	if err != nil {
		e2eutils.Failf("Failed to delete node pool %q. Err: %v\n%v", poolName, err, string(out))
	}
	e2eutils.Logf("Successfully deleted node pool %q.", poolName)

	out, err = exec.Command("gcloud", "container", "node-pools", "list",
		clusterStr).CombinedOutput()
	if err != nil {
		e2eutils.Failf("\nFailed to list node pools from cluster %q. Err: %v\n%v", e2econfig.TestContext.CloudConfig.Cluster, err, string(out))
	}
	e2eutils.Logf("\nNode pools:\n%s", string(out))

	e2eutils.Logf("Checking that no nodes have the deleted node pool's label.")
	nodeCount = nodesWithPoolLabel(f, poolName)
	if nodeCount != 0 {
		e2eutils.Failf("Wanted 0 nodes with node pool label, got: %v", nodeCount)
	}
	e2eutils.Logf("Success, found no nodes with the deleted node pool's label.")
}

// nodesWithPoolLabel returns the number of nodes that have the "gke-nodepool"
// label with the given node pool name.
func nodesWithPoolLabel(f *framework.Framework, poolName string) int {
	nodeCount := 0
	nodeList, err := e2enode.GetReadySchedulableNodes(f.ClientSet)
	e2eutils.ExpectNoError(err)
	for _, node := range nodeList.Items {
		if poolLabel := node.Labels["cloud.google.com/gke-nodepool"]; poolLabel == poolName {
			nodeCount++
		}
	}
	return nodeCount
}
