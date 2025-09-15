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

package node

import (
	"context"
	"fmt"
	"regexp"
	"time"

	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
)

const sleepTime = 20 * time.Second

var requiredPerNodePods = []*regexp.Regexp{
	regexp.MustCompile(".*kube-proxy.*"),
	regexp.MustCompile(".*fluentd-elasticsearch.*"),
	regexp.MustCompile(".*node-problem-detector.*"),
}

// WaitForReadyNodes waits up to timeout for cluster to has desired size and
// there is no not-ready nodes in it. By cluster size we mean number of schedulable Nodes.
func WaitForReadyNodes(ctx context.Context, c clientset.Interface, size int, timeout time.Duration) error {
	_, err := CheckReady(ctx, c, size, timeout)
	return err
}

// WaitForTotalHealthy checks whether all registered nodes are ready and all required Pods are running on them.
func WaitForTotalHealthy(ctx context.Context, c clientset.Interface, timeout time.Duration) error {
	framework.Logf("Waiting up to %v for all nodes to be ready", timeout)

	var notReady []v1.Node
	var missingPodsPerNode map[string][]string
	err := wait.PollUntilContextTimeout(ctx, poll, timeout, true, func(ctx context.Context) (bool, error) {
		notReady = nil
		// It should be OK to list unschedulable Nodes here.
		nodes, err := c.CoreV1().Nodes().List(ctx, metav1.ListOptions{ResourceVersion: "0"})
		if err != nil {
			return false, err
		}
		for _, node := range nodes.Items {
			if !IsConditionSetAsExpected(&node, v1.NodeReady, true) {
				notReady = append(notReady, node)
			}
		}
		pods, err := c.CoreV1().Pods(metav1.NamespaceAll).List(ctx, metav1.ListOptions{ResourceVersion: "0"})
		if err != nil {
			return false, err
		}

		systemPodsPerNode := make(map[string][]string)
		for _, pod := range pods.Items {
			if pod.Namespace == metav1.NamespaceSystem && pod.Status.Phase == v1.PodRunning {
				if pod.Spec.NodeName != "" {
					systemPodsPerNode[pod.Spec.NodeName] = append(systemPodsPerNode[pod.Spec.NodeName], pod.Name)
				}
			}
		}
		missingPodsPerNode = make(map[string][]string)
		for _, node := range nodes.Items {
			if isNodeSchedulableWithoutTaints(&node) {
				for _, requiredPod := range requiredPerNodePods {
					foundRequired := false
					for _, presentPod := range systemPodsPerNode[node.Name] {
						if requiredPod.MatchString(presentPod) {
							foundRequired = true
							break
						}
					}
					if !foundRequired {
						missingPodsPerNode[node.Name] = append(missingPodsPerNode[node.Name], requiredPod.String())
					}
				}
			}
		}
		return len(notReady) == 0 && len(missingPodsPerNode) == 0, nil
	})

	if err != nil && !wait.Interrupted(err) {
		return err
	}

	if len(notReady) > 0 {
		return fmt.Errorf("Not ready nodes: %v", notReady)
	}
	if len(missingPodsPerNode) > 0 {
		return fmt.Errorf("Not running system Pods: %v", missingPodsPerNode)
	}
	return nil

}

// WaitConditionToBe returns whether node "name's" condition state matches wantTrue
// within timeout. If wantTrue is true, it will ensure the node condition status
// is ConditionTrue; if it's false, it ensures the node condition is in any state
// other than ConditionTrue (e.g. not true or unknown).
func WaitConditionToBe(ctx context.Context, c clientset.Interface, name string, conditionType v1.NodeConditionType, wantTrue bool, timeout time.Duration) bool {
	framework.Logf("Waiting up to %v for node %s condition %s to be %t", timeout, name, conditionType, wantTrue)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(poll) {
		node, err := c.CoreV1().Nodes().Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			framework.Logf("Couldn't get node %s", name)
			continue
		}

		if IsConditionSetAsExpected(node, conditionType, wantTrue) {
			return true
		}
	}
	framework.Logf("Node %s didn't reach desired %s condition status (%t) within %v", name, conditionType, wantTrue, timeout)
	return false
}

// WaitForNodeToBeNotReady returns whether node name is not ready (i.e. the
// readiness condition is anything but ready, e.g false or unknown) within
// timeout.
func WaitForNodeToBeNotReady(ctx context.Context, c clientset.Interface, name string, timeout time.Duration) bool {
	return WaitConditionToBe(ctx, c, name, v1.NodeReady, false, timeout)
}

// WaitForNodeToBeReady returns whether node name is ready within timeout.
func WaitForNodeToBeReady(ctx context.Context, c clientset.Interface, name string, timeout time.Duration) bool {
	return WaitConditionToBe(ctx, c, name, v1.NodeReady, true, timeout)
}

func WaitForNodeSchedulable(ctx context.Context, c clientset.Interface, name string, timeout time.Duration, wantSchedulable bool) bool {
	framework.Logf("Waiting up to %v for node %s to be schedulable: %t", timeout, name, wantSchedulable)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(poll) {
		node, err := c.CoreV1().Nodes().Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			framework.Logf("Couldn't get node %s", name)
			continue
		}

		if IsNodeSchedulable(node) == wantSchedulable {
			return true
		}
	}
	framework.Logf("Node %s didn't reach desired schedulable status (%t) within %v", name, wantSchedulable, timeout)
	return false
}

// WaitForNodeHeartbeatAfter waits up to timeout for node to send the next
// heartbeat after the given timestamp.
//
// To ensure the node status is posted by a restarted kubelet process,
// after should be retrieved by [GetNodeHeartbeatTime] while the kubelet is down.
func WaitForNodeHeartbeatAfter(ctx context.Context, c clientset.Interface, name string, after metav1.Time, timeout time.Duration) {
	framework.Logf("Waiting up to %v for node %s to send a heartbeat after %v", timeout, name, after)
	gomega.Eventually(ctx, func() (time.Time, error) {
		node, err := c.CoreV1().Nodes().Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			framework.Logf("Couldn't get node %s", name)
			return time.Time{}, err
		}
		return GetNodeHeartbeatTime(node).Time, nil
	}, timeout, poll).Should(gomega.BeTemporally(">", after.Time), "Node %s didn't send a heartbeat", name)
}

// CheckReady waits up to timeout for cluster to has desired size and
// there is no not-ready nodes in it. By cluster size we mean number of schedulable Nodes.
func CheckReady(ctx context.Context, c clientset.Interface, size int, timeout time.Duration) ([]v1.Node, error) {
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(sleepTime) {
		nodes, err := waitListSchedulableNodes(ctx, c)
		if err != nil {
			framework.Logf("Failed to list nodes: %v", err)
			continue
		}
		numNodes := len(nodes.Items)

		// Filter out not-ready nodes.
		Filter(nodes, func(node v1.Node) bool {
			nodeReady := IsConditionSetAsExpected(&node, v1.NodeReady, true)
			networkReady := isConditionUnset(&node, v1.NodeNetworkUnavailable) || IsConditionSetAsExpected(&node, v1.NodeNetworkUnavailable, false)
			return nodeReady && networkReady
		})
		numReady := len(nodes.Items)

		if numNodes == size && numReady == size {
			framework.Logf("Cluster has reached the desired number of ready nodes %d", size)
			return nodes.Items, nil
		}
		framework.Logf("Waiting for ready nodes %d, current ready %d, not ready nodes %d", size, numReady, numNodes-numReady)
	}
	return nil, fmt.Errorf("timeout waiting %v for number of ready nodes to be %d", timeout, size)
}

// waitListSchedulableNodes is a wrapper around listing nodes supporting retries.
func waitListSchedulableNodes(ctx context.Context, c clientset.Interface) (*v1.NodeList, error) {
	var nodes *v1.NodeList
	var err error
	if wait.PollUntilContextTimeout(ctx, poll, singleCallTimeout, true, func(ctx context.Context) (bool, error) {
		nodes, err = c.CoreV1().Nodes().List(ctx, metav1.ListOptions{FieldSelector: fields.Set{
			"spec.unschedulable": "false",
		}.AsSelector().String()})
		if err != nil {
			return false, err
		}
		return true, nil
	}) != nil {
		return nodes, err
	}
	return nodes, nil
}

// checkWaitListSchedulableNodes is a wrapper around listing nodes supporting retries.
func checkWaitListSchedulableNodes(ctx context.Context, c clientset.Interface) (*v1.NodeList, error) {
	nodes, err := waitListSchedulableNodes(ctx, c)
	if err != nil {
		return nil, fmt.Errorf("error: %s. Non-retryable failure or timed out while listing nodes for e2e cluster", err)
	}
	return nodes, nil
}

// CheckReadyForTests returns a function which will return 'true' once the number of ready nodes is above the allowedNotReadyNodes threshold (i.e. to be used as a global gate for starting the tests).
func CheckReadyForTests(ctx context.Context, c clientset.Interface, nonblockingTaints string, allowedNotReadyNodes, largeClusterThreshold int) func(ctx context.Context) (bool, error) {
	attempt := 0
	return func(ctx context.Context) (bool, error) {
		if allowedNotReadyNodes == -1 {
			return true, nil
		}
		attempt++
		var nodesNotReadyYet []v1.Node
		opts := metav1.ListOptions{
			ResourceVersion: "0",
			// remove uncordoned nodes from our calculation, TODO refactor if node v2 API removes that semantic.
			FieldSelector: fields.Set{"spec.unschedulable": "false"}.AsSelector().String(),
		}
		allNodes, err := c.CoreV1().Nodes().List(ctx, opts)
		if err != nil {
			var terminalListNodesErr error
			framework.Logf("Unexpected error listing nodes: %v", err)
			if attempt >= 3 {
				terminalListNodesErr = err
			}
			return false, terminalListNodesErr
		}
		for _, node := range allNodes.Items {
			if !readyForTests(&node, nonblockingTaints) {
				nodesNotReadyYet = append(nodesNotReadyYet, node)
			}
		}
		// Framework allows for <TestContext.AllowedNotReadyNodes> nodes to be non-ready,
		// to make it possible e.g. for incorrect deployment of some small percentage
		// of nodes (which we allow in cluster validation). Some nodes that are not
		// provisioned correctly at startup will never become ready (e.g. when something
		// won't install correctly), so we can't expect them to be ready at any point.
		//
		// We log the *reason* why nodes are not schedulable, specifically, its usually the network not being available.
		if len(nodesNotReadyYet) > 0 {
			// In large clusters, log them only every 10th pass.
			if len(nodesNotReadyYet) < largeClusterThreshold || attempt%10 == 0 {
				framework.Logf("Unschedulable nodes= %v, maximum value for starting tests= %v", len(nodesNotReadyYet), allowedNotReadyNodes)
				for _, node := range nodesNotReadyYet {
					framework.Logf("	-> Node %s [[[ Ready=%t, Network(available)=%t, Taints=%v, NonblockingTaints=%v ]]]",
						node.Name,
						IsConditionSetAsExpectedSilent(&node, v1.NodeReady, true),
						IsConditionSetAsExpectedSilent(&node, v1.NodeNetworkUnavailable, false),
						node.Spec.Taints,
						nonblockingTaints,
					)

				}
				if len(nodesNotReadyYet) > allowedNotReadyNodes {
					ready := len(allNodes.Items) - len(nodesNotReadyYet)
					remaining := len(nodesNotReadyYet) - allowedNotReadyNodes
					framework.Logf("==== node wait: %v out of %v nodes are ready, max notReady allowed %v.  Need %v more before starting.", ready, len(allNodes.Items), allowedNotReadyNodes, remaining)
				}
			}
		}
		return len(nodesNotReadyYet) <= allowedNotReadyNodes, nil
	}
}

// readyForTests determines whether or not we should continue waiting for the nodes
// to enter a testable state. By default this means it is schedulable, NodeReady, and untainted.
// Nodes with taints nonblocking taints are permitted to have that taint and
// also have their node.Spec.Unschedulable field ignored for the purposes of this function.
func readyForTests(node *v1.Node, nonblockingTaints string) bool {
	if hasNonblockingTaint(node, nonblockingTaints) {
		// If the node has one of the nonblockingTaints taints; just check that it is ready
		// and don't require node.Spec.Unschedulable to be set either way.
		if !IsNodeReady(node) || !isNodeUntaintedWithNonblocking(node, nonblockingTaints) {
			return false
		}
	} else {
		if !IsNodeSchedulable(node) || !isNodeUntainted(node) {
			return false
		}
	}
	return true
}
