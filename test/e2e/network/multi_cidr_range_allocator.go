/*
Copyright 2023 The Kubernetes Authors.

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

package network

import (
	"context"
	"net"
	"time"

	"github.com/onsi/ginkgo/v2"
	v1 "k8s.io/api/core/v1"
	v1alpha1 "k8s.io/api/networking/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/network/common"
	admissionapi "k8s.io/pod-security-admission/api"
	netutils "k8s.io/utils/net"
)

var _ = common.SIGDescribe("[Feature:MultiCIDRRangeAllocator][Disruptive]", func() {
	f := framework.NewDefaultFramework("multicidrrangeallocator")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

	var cs clientset.Interface
	var cidr *net.IPNet
	var clusterCIDR *v1alpha1.ClusterCIDR
	var targetNode, nodeToCreate *v1.Node
	var err error

	type Action int
	const (
		CreateNodeObj Action = iota
		DeleteNodeObj
		ValidateNodeDeleted
		CreateClusterCIDRObj
		DeleteClusterCIDRObj
		ValidatePodCIDR
	)

	step := func(ctx context.Context, action Action) {
		switch action {
		case CreateNodeObj:
			nodeToCreate.ObjectMeta.SetResourceVersion("0")
			nodeToCreate.Spec.PodCIDR = ""
			nodeToCreate.Spec.PodCIDRs = []string{}
			_, err = cs.CoreV1().Nodes().Create(ctx, nodeToCreate, metav1.CreateOptions{})
			if err != nil {
				framework.Failf("failed to create node %q, err: %q", nodeToCreate.Name, err)
			}
		case DeleteNodeObj:
			err = cs.CoreV1().Nodes().Delete(ctx, targetNode.Name, metav1.DeleteOptions{})
			if err != nil {
				framework.Failf("failed to delete node %q, err: %q", targetNode.Name, err)
			}
		case ValidateNodeDeleted:
			_, err = cs.CoreV1().Nodes().Get(ctx, targetNode.Name, metav1.GetOptions{})
			if err == nil {
				framework.Failf("node %q still exists when it should be deleted", targetNode.Name)
			} else if !apierrors.IsNotFound(err) {
				framework.Failf("failed to get node %q err: %q", targetNode.Name, err)
			}
		case CreateClusterCIDRObj:
			_, err = cs.NetworkingV1alpha1().ClusterCIDRs().Create(ctx, clusterCIDR, metav1.CreateOptions{})
			if err != nil {
				framework.Failf("failed to create clusterCIDRs %q for node %q, err: %q", clusterCIDR.Name, targetNode.Name, err)
			}
		case DeleteClusterCIDRObj:
			err = cs.NetworkingV1alpha1().ClusterCIDRs().Delete(ctx, clusterCIDR.Name, metav1.DeleteOptions{})
			if err != nil {
				framework.Failf("failed to delete clusterCIDR %q for %q, err: %q", clusterCIDR.Name, targetNode.Name, err)
			}
		case ValidatePodCIDR:
			createdNode, err := cs.CoreV1().Nodes().Get(ctx, nodeToCreate.Name, metav1.GetOptions{})
			if err != nil {
				framework.Failf("failed to get node %q err: %q", nodeToCreate.Name, err)
			}
			_, assignedCIDR, err := netutils.ParseCIDRSloppy(createdNode.Spec.PodCIDR)
			if err != nil {
				framework.Failf("failed to parse CIDR %q err: %q", createdNode.Spec.PodCIDR, err)
			}
			if !containsCIDR(cidr, assignedCIDR) {
				framework.Failf("assigned Pod CIDR %q is not from the expected configured ClusterCIDR %q", createdNode.Spec.PodCIDR, cidr.String())
			}
		}
	}

	ginkgo.BeforeEach(func(ctx context.Context) {
		// Only supported in GCE because those are the only cloud providers
		// where E2E test are currently running.
		e2eskipper.SkipUnlessProviderIs("gce")
		cs = f.ClientSet

		// Wait for the nodes to be ready.
		framework.ExpectNoError(e2enode.AllNodesReady(ctx, cs, 5*time.Minute))
	})

	ginkgo.AfterEach(func(ctx context.Context) {
		// Clean up the ClusterCIDR object. A cluster CIDR cannot be deleted unless
		// all the associated nodes are deleted. Thus delete the node, delete the
		// ClusterCIDR object and create the node again.
		step(ctx, DeleteNodeObj)

		// Wait for the nodes to be ready.
		framework.ExpectNoError(e2enode.AllNodesReady(ctx, cs, 5*time.Minute))

		step(ctx, ValidateNodeDeleted)

		step(ctx, DeleteClusterCIDRObj)

		step(ctx, CreateNodeObj)

		// Wait for the nodes to be ready.
		framework.ExpectNoError(e2enode.AllNodesReady(ctx, cs, 5*time.Minute))
	})

	ginkgo.It("should configure Pod CIDRs from a newly added discontiguous range", func(ctx context.Context) {
		targetNode, err = e2enode.GetRandomReadySchedulableNode(ctx, cs)
		nodeToCreate = targetNode.DeepCopy()
		framework.ExpectNoError(err)

		clusterCIDRName := "test-new-node-pod-cidr"

		// Configure a Discontiguous CIDR range(10.245.0.0/16), the nodes were
		// initially assigned Pod CIDRs from the default CIDR range(10.64.0.0/14)
		_, cidr, _ = netutils.ParseCIDRSloppy("10.245.0.0/16")
		clusterCIDR = clusterCIDRObj(clusterCIDRName, cidr.String(), targetNode.Name)
		// Create a ClusterCIDR object.
		step(ctx, CreateClusterCIDRObj)

		// Delete the Node object.
		step(ctx, DeleteNodeObj)

		// Wait for the nodes to be ready.
		framework.ExpectNoError(e2enode.AllNodesReady(ctx, cs, 5*time.Minute))

		// Make sure that the node is deleted
		step(ctx, ValidateNodeDeleted)

		// Create the Node Object again, so that it gets the new CIDR assigned.
		step(ctx, CreateNodeObj)

		// Wait for the nodes to be ready.
		framework.ExpectNoError(e2enode.AllNodesReady(ctx, cs, 5*time.Minute))

		// Validate that the New Node object has Pod CIDRs assigned from the created
		// ClusterCIDR.
		step(ctx, ValidatePodCIDR)
	})
})

func clusterCIDRObj(clusterCIDRName, cidr, hostname string) *v1alpha1.ClusterCIDR {
	return &v1alpha1.ClusterCIDR{
		ObjectMeta: metav1.ObjectMeta{
			Name: clusterCIDRName,
		},
		Spec: v1alpha1.ClusterCIDRSpec{
			IPv4:            cidr,
			PerNodeHostBits: 8,
			NodeSelector: &v1.NodeSelector{
				NodeSelectorTerms: []v1.NodeSelectorTerm{
					{
						MatchExpressions: []v1.NodeSelectorRequirement{
							{
								Key:      "kubernetes.io/hostname",
								Operator: "In",
								Values:   []string{hostname},
							},
						},
					},
				},
			},
		},
	}
}

// containsCIDR returns true if outer contains inner.
func containsCIDR(outer, inner *net.IPNet) bool {
	return outer.Contains(firstIPInRange(inner)) && outer.Contains(lastIPInRange(inner))
}

// firstIPInRange returns the first IP in a given IP range.
func firstIPInRange(ipNet *net.IPNet) net.IP {
	return ipNet.IP.Mask(ipNet.Mask)
}

// lastIPInRange returns the last IP in a given IP range.
func lastIPInRange(cidr *net.IPNet) net.IP {
	ip := append([]byte{}, cidr.IP...)
	for i, b := range cidr.Mask {
		ip[i] |= ^b
	}
	return ip
}
