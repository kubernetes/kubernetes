/*
Copyright 2014 The Kubernetes Authors.

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

package node

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/util/retry"

	"k8s.io/kubernetes/test/e2e/framework"
	testutils "k8s.io/kubernetes/test/utils"
)

const (
	// Minimal number of nodes for the cluster to be considered large.
	largeClusterThreshold = 100
)

// WaitForAllNodesSchedulable waits up to timeout for all
// (but TestContext.AllowedNotReadyNodes) to become schedulable.
func WaitForAllNodesSchedulable(ctx context.Context, c clientset.Interface, timeout time.Duration) error {
	if framework.TestContext.AllowedNotReadyNodes == -1 {
		return nil
	}

	framework.Logf("Waiting up to %v for all (but %d) nodes to be schedulable", timeout, framework.TestContext.AllowedNotReadyNodes)
	return wait.PollUntilContextTimeout(
		ctx,
		30*time.Second,
		timeout,
		true,
		CheckReadyForTests(ctx, c, framework.TestContext.NonblockingTaints, framework.TestContext.AllowedNotReadyNodes, largeClusterThreshold),
	)
}

// AddOrUpdateLabelOnNode adds the given label key and value to the given node or updates value.
func AddOrUpdateLabelOnNode(c clientset.Interface, nodeName string, labelKey, labelValue string) {
	framework.ExpectNoError(testutils.AddLabelsToNode(c, nodeName, map[string]string{labelKey: labelValue}))
}

// ExpectNodeHasLabel expects that the given node has the given label pair.
func ExpectNodeHasLabel(ctx context.Context, c clientset.Interface, nodeName string, labelKey string, labelValue string) {
	ginkgo.By("verifying the node has the label " + labelKey + " " + labelValue)
	node, err := c.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
	framework.ExpectNoError(err)
	gomega.Expect(node.Labels).To(gomega.HaveKeyWithValue(labelKey, labelValue))
}

// RemoveLabelOffNode is for cleaning up labels temporarily added to node,
// won't fail if target label doesn't exist or has been removed.
func RemoveLabelOffNode(c clientset.Interface, nodeName string, labelKey string) {
	ginkgo.By("removing the label " + labelKey + " off the node " + nodeName)
	framework.ExpectNoError(testutils.RemoveLabelOffNode(c, nodeName, []string{labelKey}))

	ginkgo.By("verifying the node doesn't have the label " + labelKey)
	framework.ExpectNoError(testutils.VerifyLabelsRemoved(c, nodeName, []string{labelKey}))
}

// ExpectNodeHasTaint expects that the node has the given taint.
func ExpectNodeHasTaint(ctx context.Context, c clientset.Interface, nodeName string, taint *v1.Taint) {
	ginkgo.By("verifying the node has the taint " + taint.ToString())
	if has, err := NodeHasTaint(ctx, c, nodeName, taint); !has {
		framework.ExpectNoError(err)
		framework.Failf("Failed to find taint %s on node %s", taint.ToString(), nodeName)
	}
}

// NodeHasTaint returns true if the node has the given taint, else returns false.
func NodeHasTaint(ctx context.Context, c clientset.Interface, nodeName string, taint *v1.Taint) (bool, error) {
	node, err := c.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
	if err != nil {
		return false, err
	}

	nodeTaints := node.Spec.Taints

	if len(nodeTaints) == 0 || !taintExists(nodeTaints, taint) {
		return false, nil
	}
	return true, nil
}

// AllNodesReady checks whether all registered nodes are ready. Setting -1 on
// framework.TestContext.AllowedNotReadyNodes will bypass the post test node readiness check.
// TODO: we should change the AllNodesReady call in AfterEach to WaitForAllNodesHealthy,
// and figure out how to do it in a configurable way, as we can't expect all setups to run
// default test add-ons.
func AllNodesReady(ctx context.Context, c clientset.Interface, timeout time.Duration) error {
	if err := allNodesReady(ctx, c, timeout); err != nil {
		return fmt.Errorf("checking for ready nodes: %w", err)
	}
	return nil
}

func allNodesReady(ctx context.Context, c clientset.Interface, timeout time.Duration) error {
	if framework.TestContext.AllowedNotReadyNodes == -1 {
		return nil
	}

	framework.Logf("Waiting up to %v for all (but %d) nodes to be ready", timeout, framework.TestContext.AllowedNotReadyNodes)

	var notReady []*v1.Node
	err := wait.PollUntilContextTimeout(ctx, framework.Poll, timeout, true, func(ctx context.Context) (bool, error) {
		notReady = nil
		// It should be OK to list unschedulable Nodes here.
		nodes, err := c.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
		if err != nil {
			return false, err
		}
		for i := range nodes.Items {
			node := &nodes.Items[i]
			if !IsConditionSetAsExpected(node, v1.NodeReady, true) {
				notReady = append(notReady, node)
			}
		}
		// Framework allows for <TestContext.AllowedNotReadyNodes> nodes to be non-ready,
		// to make it possible e.g. for incorrect deployment of some small percentage
		// of nodes (which we allow in cluster validation). Some nodes that are not
		// provisioned correctly at startup will never become ready (e.g. when something
		// won't install correctly), so we can't expect them to be ready at any point.
		return len(notReady) <= framework.TestContext.AllowedNotReadyNodes, nil
	})

	if err != nil && !wait.Interrupted(err) {
		return err
	}

	if len(notReady) > framework.TestContext.AllowedNotReadyNodes {
		msg := ""
		for _, node := range notReady {
			msg = fmt.Sprintf("%s, %s", msg, node.Name)
		}
		return fmt.Errorf("Not ready nodes: %#v", msg)
	}
	return nil
}

// taintExists checks if the given taint exists in list of taints. Returns true if exists false otherwise.
func taintExists(taints []v1.Taint, taintToFind *v1.Taint) bool {
	for _, taint := range taints {
		if taint.MatchTaint(taintToFind) {
			return true
		}
	}
	return false
}

// IsARM64 checks whether the k8s Node has arm64 arch.
func IsARM64(node *v1.Node) bool {
	arch, ok := node.Labels["kubernetes.io/arch"]
	if ok {
		return arch == "arm64"
	}

	return false
}

// AddExtendedResource adds a fake resource to the Node.
func AddExtendedResource(ctx context.Context, clientSet clientset.Interface, nodeName string, extendedResourceName v1.ResourceName, extendedResourceQuantity resource.Quantity) {
	extendedResource := v1.ResourceName(extendedResourceName)

	ginkgo.By("Adding a custom resource")
	extendedResourceList := v1.ResourceList{
		extendedResource: extendedResourceQuantity,
	}

	// This is a workaround for the fact that we shouldn't marshal a Node struct to JSON
	// because it wipes out some fields from node status like the daemonEndpoints and
	// nodeInfo which should not be changed at this time. We need to use a map instead.
	// See https://github.com/kubernetes/kubernetes/issues/131229
	patchPayload, err := json.Marshal(map[string]any{
		"status": map[string]any{
			"capacity":    extendedResourceList,
			"allocatable": extendedResourceList,
		},
	})
	framework.ExpectNoError(err, "Failed to marshal node JSON")

	_, err = clientSet.CoreV1().Nodes().Patch(ctx, nodeName, types.StrategicMergePatchType, []byte(patchPayload), metav1.PatchOptions{}, "status")
	framework.ExpectNoError(err)
}

// RemoveExtendedResource removes a fake resource from the Node.
func RemoveExtendedResource(ctx context.Context, clientSet clientset.Interface, nodeName string, extendedResourceName v1.ResourceName) {
	extendedResource := v1.ResourceName(extendedResourceName)

	ginkgo.By("Removing a custom resource")
	err := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		node, err := clientSet.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
		if err != nil {
			return fmt.Errorf("failed to get node %s: %w", nodeName, err)
		}
		delete(node.Status.Capacity, extendedResource)
		delete(node.Status.Allocatable, extendedResource)
		_, err = clientSet.CoreV1().Nodes().UpdateStatus(ctx, node, metav1.UpdateOptions{})
		return err
	})
	framework.ExpectNoError(err)
}
