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
	"context"
	"fmt"
	"os/exec"

	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
)

var _ = SIGDescribe("GKE node pools", feature.GKENodePool, func() {

	f := framework.NewDefaultFramework("node-pools")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.BeforeEach(func() {
		e2eskipper.SkipUnlessProviderIs("gce", "gke")
	})

	f.It("should create a cluster with multiple node pools", feature.GKENodePool, func(ctx context.Context) {
		framework.Logf("Start create node pool test")
		testCreateDeleteNodePool(ctx, f, "test-pool")
	})
})

func testCreateDeleteNodePool(ctx context.Context, f *framework.Framework, poolName string) {
	framework.Logf("Create node pool: %q in cluster: %q", poolName, framework.TestContext.CloudConfig.Cluster)

	clusterStr := fmt.Sprintf("--cluster=%s", framework.TestContext.CloudConfig.Cluster)

	out, err := exec.Command("gcloud", "container", "node-pools", "create",
		poolName,
		clusterStr,
		"--num-nodes=2").CombinedOutput()
	framework.Logf("\n%s", string(out))
	if err != nil {
		framework.Failf("Failed to create node pool %q. Err: %v\n%v", poolName, err, string(out))
	}
	framework.Logf("Successfully created node pool %q.", poolName)

	out, err = exec.Command("gcloud", "container", "node-pools", "list",
		clusterStr).CombinedOutput()
	if err != nil {
		framework.Failf("Failed to list node pools from cluster %q. Err: %v\n%v", framework.TestContext.CloudConfig.Cluster, err, string(out))
	}
	framework.Logf("Node pools:\n%s", string(out))

	framework.Logf("Checking that 2 nodes have the correct node pool label.")
	nodeCount := nodesWithPoolLabel(ctx, f, poolName)
	if nodeCount != 2 {
		framework.Failf("Wanted 2 nodes with node pool label, got: %v", nodeCount)
	}
	framework.Logf("Success, found 2 nodes with correct node pool labels.")

	framework.Logf("Deleting node pool: %q in cluster: %q", poolName, framework.TestContext.CloudConfig.Cluster)
	out, err = exec.Command("gcloud", "container", "node-pools", "delete",
		poolName,
		clusterStr,
		"-q").CombinedOutput()
	framework.Logf("\n%s", string(out))
	if err != nil {
		framework.Failf("Failed to delete node pool %q. Err: %v\n%v", poolName, err, string(out))
	}
	framework.Logf("Successfully deleted node pool %q.", poolName)

	out, err = exec.Command("gcloud", "container", "node-pools", "list",
		clusterStr).CombinedOutput()
	if err != nil {
		framework.Failf("\nFailed to list node pools from cluster %q. Err: %v\n%v", framework.TestContext.CloudConfig.Cluster, err, string(out))
	}
	framework.Logf("\nNode pools:\n%s", string(out))

	framework.Logf("Checking that no nodes have the deleted node pool's label.")
	nodeCount = nodesWithPoolLabel(ctx, f, poolName)
	if nodeCount != 0 {
		framework.Failf("Wanted 0 nodes with node pool label, got: %v", nodeCount)
	}
	framework.Logf("Success, found no nodes with the deleted node pool's label.")
}

// nodesWithPoolLabel returns the number of nodes that have the "gke-nodepool"
// label with the given node pool name.
func nodesWithPoolLabel(ctx context.Context, f *framework.Framework, poolName string) int {
	nodeCount := 0
	nodeList, err := e2enode.GetReadySchedulableNodes(ctx, f.ClientSet)
	framework.ExpectNoError(err)
	for _, node := range nodeList.Items {
		if poolLabel := node.Labels["cloud.google.com/gke-nodepool"]; poolLabel == poolName {
			nodeCount++
		}
	}
	return nodeCount
}
