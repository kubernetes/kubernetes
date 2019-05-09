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
	"time"

	apierrs "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

var _ = SIGDescribe("[Feature:CloudProvider][Disruptive] Nodes", func() {
	f := framework.NewDefaultFramework("cloudprovider")
	var c clientset.Interface

	ginkgo.BeforeEach(func() {
		// Only supported in AWS/GCE because those are the only cloud providers
		// where E2E test are currently running.
		framework.SkipUnlessProviderIs("aws", "gce", "gke")
		c = f.ClientSet
	})

	ginkgo.It("should be deleted on API server if it doesn't exist in the cloud provider", func() {
		ginkgo.By("deleting a node on the cloud provider")

		nodeDeleteCandidates := framework.GetReadySchedulableNodesOrDie(c)
		nodeToDelete := nodeDeleteCandidates.Items[0]

		origNodes := framework.GetReadyNodesIncludingTaintedOrDie(c)
		framework.Logf("Original number of ready nodes: %d", len(origNodes.Items))

		err := framework.DeleteNodeOnCloudProvider(&nodeToDelete)
		if err != nil {
			framework.Failf("failed to delete node %q, err: %q", nodeToDelete.Name, err)
		}

		newNodes, err := framework.CheckNodesReady(c, len(origNodes.Items)-1, 5*time.Minute)
		gomega.Expect(err).To(gomega.BeNil())
		gomega.Expect(len(newNodes)).To(gomega.Equal(len(origNodes.Items) - 1))

		_, err = c.CoreV1().Nodes().Get(nodeToDelete.Name, metav1.GetOptions{})
		if err == nil {
			framework.Failf("node %q still exists when it should be deleted", nodeToDelete.Name)
		} else if !apierrs.IsNotFound(err) {
			framework.Failf("failed to get node %q err: %q", nodeToDelete.Name, err)
		}

	})
})
