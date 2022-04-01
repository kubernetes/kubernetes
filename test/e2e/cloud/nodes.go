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

package cloud

import (
	"context"
	"time"

	e2econfig "k8s.io/kubernetes/test/e2e/framework/config"
	e2eutils "k8s.io/kubernetes/test/e2e/framework/utils"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"

	"github.com/onsi/ginkgo"
)

var _ = SIGDescribe("[Feature:CloudProvider][Disruptive] Nodes", func() {
	f := framework.NewDefaultFramework("cloudprovider")
	var c clientset.Interface

	ginkgo.BeforeEach(func() {
		// Only supported in AWS/GCE because those are the only cloud providers
		// where E2E test are currently running.
		e2eskipper.SkipUnlessProviderIs("aws", "gce", "gke")
		c = f.ClientSet
	})

	ginkgo.It("should be deleted on API server if it doesn't exist in the cloud provider", func() {
		ginkgo.By("deleting a node on the cloud provider")

		nodeToDelete, err := e2enode.GetRandomReadySchedulableNode(c)
		e2eutils.ExpectNoError(err)

		origNodes, err := e2enode.GetReadyNodesIncludingTainted(c)
		if err != nil {
			e2eutils.Logf("Unexpected error occurred: %v", err)
		}
		e2eutils.ExpectNoErrorWithOffset(0, err)

		e2eutils.Logf("Original number of ready nodes: %d", len(origNodes.Items))

		err = deleteNodeOnCloudProvider(nodeToDelete)
		if err != nil {
			e2eutils.Failf("failed to delete node %q, err: %q", nodeToDelete.Name, err)
		}

		newNodes, err := e2enode.CheckReady(c, len(origNodes.Items)-1, 5*time.Minute)
		e2eutils.ExpectNoError(err)
		e2eutils.ExpectEqual(len(newNodes), len(origNodes.Items)-1)

		_, err = c.CoreV1().Nodes().Get(context.TODO(), nodeToDelete.Name, metav1.GetOptions{})
		if err == nil {
			e2eutils.Failf("node %q still exists when it should be deleted", nodeToDelete.Name)
		} else if !apierrors.IsNotFound(err) {
			e2eutils.Failf("failed to get node %q err: %q", nodeToDelete.Name, err)
		}

	})
})

// DeleteNodeOnCloudProvider deletes the specified node.
func deleteNodeOnCloudProvider(node *v1.Node) error {
	return e2econfig.TestContext.CloudConfig.Provider.DeleteNode(node)
}
