/*
Copyright 2018 The Kubernetes Authors.

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
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var _ = SIGDescribe(framework.WithDisruptive(), "NodeLease", func() {
	f := framework.NewDefaultFramework("node-lease-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	var systemPodsNo int
	var c clientset.Interface
	var ns string
	var group string

	ginkgo.BeforeEach(func(ctx context.Context) {
		e2eskipper.SkipUnlessProviderIs("gce")
		c = f.ClientSet
		ns = f.Namespace.Name
		systemPods, err := e2epod.GetPodsInNamespace(ctx, c, ns, map[string]string{})
		framework.ExpectNoError(err)
		systemPodsNo = len(systemPods)
		if strings.Contains(framework.TestContext.CloudConfig.NodeInstanceGroup, ",") {
			framework.Failf("Test dose not support cluster setup with more than one MIG: %s", framework.TestContext.CloudConfig.NodeInstanceGroup)
		} else {
			group = framework.TestContext.CloudConfig.NodeInstanceGroup
		}
	})

	ginkgo.Describe("NodeLease deletion", func() {
		var skipped bool

		ginkgo.BeforeEach(func() {
			skipped = true
			e2eskipper.SkipUnlessProviderIs("gce", "aws")
			e2eskipper.SkipUnlessNodeCountIsAtLeast(2)
			skipped = false
		})

		ginkgo.AfterEach(func(ctx context.Context) {
			if skipped {
				return
			}

			ginkgo.By("restoring the original node instance group size")
			if err := framework.ResizeGroup(group, int32(framework.TestContext.CloudConfig.NumNodes)); err != nil {
				framework.Failf("Couldn't restore the original node instance group size: %v", err)
			}

			if err := framework.WaitForGroupSize(group, int32(framework.TestContext.CloudConfig.NumNodes)); err != nil {
				framework.Failf("Couldn't restore the original node instance group size: %v", err)
			}

			if err := e2enode.WaitForReadyNodes(ctx, c, framework.TestContext.CloudConfig.NumNodes, 10*time.Minute); err != nil {
				framework.Failf("Couldn't restore the original cluster size: %v", err)
			}
			// Many e2e tests assume that the cluster is fully healthy before they start.  Wait until
			// the cluster is restored to health.
			ginkgo.By("waiting for system pods to successfully restart")
			err := e2epod.WaitForPodsRunningReady(ctx, c, metav1.NamespaceSystem, systemPodsNo, framework.PodReadyBeforeTimeout)
			framework.ExpectNoError(err)
		})

		ginkgo.It("node lease should be deleted when corresponding node is deleted", func(ctx context.Context) {
			leaseClient := c.CoordinationV1().Leases(v1.NamespaceNodeLease)
			err := e2enode.WaitForReadyNodes(ctx, c, framework.TestContext.CloudConfig.NumNodes, 10*time.Minute)
			framework.ExpectNoError(err)

			ginkgo.By("verify node lease exists for every nodes")
			originalNodes, err := e2enode.GetReadySchedulableNodes(ctx, c)
			framework.ExpectNoError(err)
			gomega.Expect(originalNodes.Items).To(gomega.HaveLen(framework.TestContext.CloudConfig.NumNodes))

			gomega.Eventually(ctx, func() error {
				pass := true
				for _, node := range originalNodes.Items {
					if _, err := leaseClient.Get(ctx, node.ObjectMeta.Name, metav1.GetOptions{}); err != nil {
						framework.Logf("Try to get lease of node %s, but got error: %v", node.ObjectMeta.Name, err)
						pass = false
					}
				}
				if pass {
					return nil
				}
				return fmt.Errorf("some node lease is not ready")
			}, 1*time.Minute, 5*time.Second).Should(gomega.BeNil())

			targetNumNodes := int32(framework.TestContext.CloudConfig.NumNodes - 1)
			ginkgo.By(fmt.Sprintf("decreasing cluster size to %d", targetNumNodes))
			err = framework.ResizeGroup(group, targetNumNodes)
			framework.ExpectNoError(err)
			err = framework.WaitForGroupSize(group, targetNumNodes)
			framework.ExpectNoError(err)
			err = e2enode.WaitForReadyNodes(ctx, c, framework.TestContext.CloudConfig.NumNodes-1, 10*time.Minute)
			framework.ExpectNoError(err)
			targetNodes, err := e2enode.GetReadySchedulableNodes(ctx, c)
			framework.ExpectNoError(err)
			gomega.Expect(targetNodes.Items).To(gomega.HaveLen(int(targetNumNodes)))

			ginkgo.By("verify node lease is deleted for the deleted node")
			var deletedNodeName string
			for _, originalNode := range originalNodes.Items {
				originalNodeName := originalNode.ObjectMeta.Name
				var found bool
				for _, targetNode := range targetNodes.Items {
					if originalNodeName == targetNode.ObjectMeta.Name {
						found = true
						break
					}
				}
				if !found {
					deletedNodeName = originalNodeName
					break
				}
			}
			gomega.Expect(deletedNodeName).NotTo(gomega.BeEmpty())
			gomega.Eventually(ctx, func() error {
				if _, err := leaseClient.Get(ctx, deletedNodeName, metav1.GetOptions{}); err == nil {
					return fmt.Errorf("node lease is not deleted yet for node %q", deletedNodeName)
				}
				return nil
			}, 1*time.Minute, 5*time.Second).Should(gomega.BeNil())

			ginkgo.By("verify node leases still exist for remaining nodes")
			gomega.Eventually(ctx, func() error {
				for _, node := range targetNodes.Items {
					if _, err := leaseClient.Get(ctx, node.ObjectMeta.Name, metav1.GetOptions{}); err != nil {
						return err
					}
				}
				return nil
			}, 1*time.Minute, 5*time.Second).Should(gomega.BeNil())
		})
	})
})
