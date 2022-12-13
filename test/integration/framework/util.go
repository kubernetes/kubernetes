/*
Copyright 2017 The Kubernetes Authors.

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

// TODO: This file can potentially be moved to a common place used by both e2e and integration tests.

package framework

import (
	"context"
	"fmt"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
	nodectlr "k8s.io/kubernetes/pkg/controller/nodelifecycle"
)

const (
	// poll is how often to Poll pods, nodes and claims.
	poll = 2 * time.Second

	// singleCallTimeout is how long to try single API calls (like 'get' or 'list'). Used to prevent
	// transient failures from failing tests.
	singleCallTimeout = 5 * time.Minute
)

// CreateNamespaceOrDie creates a namespace.
func CreateNamespaceOrDie(c clientset.Interface, baseName string, t testing.TB) *v1.Namespace {
	ns := &v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: baseName}}
	result, err := c.CoreV1().Namespaces().Create(context.TODO(), ns, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create namespace: %v", err)
	}
	return result
}

// DeleteNamespaceOrDie deletes a namespace.
func DeleteNamespaceOrDie(c clientset.Interface, ns *v1.Namespace, t testing.TB) {
	err := c.CoreV1().Namespaces().Delete(context.TODO(), ns.Name, metav1.DeleteOptions{})
	if err != nil {
		t.Fatalf("Failed to delete namespace: %v", err)
	}
}

// waitListAllNodes is a wrapper around listing nodes supporting retries.
func waitListAllNodes(c clientset.Interface) (*v1.NodeList, error) {
	var nodes *v1.NodeList
	var err error
	if wait.PollImmediate(poll, singleCallTimeout, func() (bool, error) {
		nodes, err = c.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{})
		if err != nil {
			return false, err
		}
		return true, nil
	}) != nil {
		return nodes, err
	}
	return nodes, nil
}

// Filter filters nodes in NodeList in place, removing nodes that do not
// satisfy the given condition
func Filter(nodeList *v1.NodeList, fn func(node v1.Node) bool) {
	var l []v1.Node

	for _, node := range nodeList.Items {
		if fn(node) {
			l = append(l, node)
		}
	}
	nodeList.Items = l
}

// IsNodeSchedulable returns true if:
// 1) doesn't have "unschedulable" field set
// 2) it also returns true from IsNodeReady
func IsNodeSchedulable(node *v1.Node) bool {
	if node == nil {
		return false
	}
	return !node.Spec.Unschedulable && IsNodeReady(node)
}

// IsNodeReady returns true if:
// 1) it's Ready condition is set to true
// 2) doesn't have NetworkUnavailable condition set to true
func IsNodeReady(node *v1.Node) bool {
	nodeReady := IsConditionSetAsExpected(node, v1.NodeReady, true)
	networkReady := isConditionUnset(node, v1.NodeNetworkUnavailable) ||
		IsConditionSetAsExpectedSilent(node, v1.NodeNetworkUnavailable, false)
	return nodeReady && networkReady
}

// IsConditionSetAsExpected returns a wantTrue value if the node has a match to the conditionType, otherwise returns an opposite value of the wantTrue with detailed logging.
func IsConditionSetAsExpected(node *v1.Node, conditionType v1.NodeConditionType, wantTrue bool) bool {
	return isNodeConditionSetAsExpected(node, conditionType, wantTrue, false)
}

// IsConditionSetAsExpectedSilent returns a wantTrue value if the node has a match to the conditionType, otherwise returns an opposite value of the wantTrue.
func IsConditionSetAsExpectedSilent(node *v1.Node, conditionType v1.NodeConditionType, wantTrue bool) bool {
	return isNodeConditionSetAsExpected(node, conditionType, wantTrue, true)
}

// isConditionUnset returns true if conditions of the given node do not have a match to the given conditionType, otherwise false.
func isConditionUnset(node *v1.Node, conditionType v1.NodeConditionType) bool {
	for _, cond := range node.Status.Conditions {
		if cond.Type == conditionType {
			return false
		}
	}
	return true
}

// isNodeConditionSetAsExpected checks a node for a condition, and returns 'true' if the wanted value is the same as the condition value, useful for polling until a condition on a node is met.
func isNodeConditionSetAsExpected(node *v1.Node, conditionType v1.NodeConditionType, wantTrue, silent bool) bool {
	// Check the node readiness condition (logging all).
	for _, cond := range node.Status.Conditions {
		// Ensure that the condition type and the status matches as desired.
		if cond.Type == conditionType {
			// For NodeReady condition we need to check Taints as well
			if cond.Type == v1.NodeReady {
				hasNodeControllerTaints := false
				// For NodeReady we need to check if Taints are gone as well
				taints := node.Spec.Taints
				for _, taint := range taints {
					if taint.MatchTaint(nodectlr.UnreachableTaintTemplate) || taint.MatchTaint(nodectlr.NotReadyTaintTemplate) {
						hasNodeControllerTaints = true
						break
					}
				}
				if wantTrue {
					if (cond.Status == v1.ConditionTrue) && !hasNodeControllerTaints {
						return true
					}
					msg := ""
					if !hasNodeControllerTaints {
						msg = fmt.Sprintf("Condition %s of node %s is %v instead of %t. Reason: %v, message: %v",
							conditionType, node.Name, cond.Status == v1.ConditionTrue, wantTrue, cond.Reason, cond.Message)
					} else {
						msg = fmt.Sprintf("Condition %s of node %s is %v, but Node is tainted by NodeController with %v. Failure",
							conditionType, node.Name, cond.Status == v1.ConditionTrue, taints)
					}
					if !silent {
						klog.Infof(msg)
					}
					return false
				}
				// TODO: check if the Node is tainted once we enable NC notReady/unreachable taints by default
				if cond.Status != v1.ConditionTrue {
					return true
				}
				if !silent {
					klog.Infof("Condition %s of node %s is %v instead of %t. Reason: %v, message: %v",
						conditionType, node.Name, cond.Status == v1.ConditionTrue, wantTrue, cond.Reason, cond.Message)
				}
				return false
			}
			if (wantTrue && (cond.Status == v1.ConditionTrue)) || (!wantTrue && (cond.Status != v1.ConditionTrue)) {
				return true
			}
			if !silent {
				klog.Infof("Condition %s of node %s is %v instead of %t. Reason: %v, message: %v",
					conditionType, node.Name, cond.Status == v1.ConditionTrue, wantTrue, cond.Reason, cond.Message)
			}
			return false
		}

	}
	if !silent {
		klog.Infof("Couldn't find condition %v on node %v", conditionType, node.Name)
	}
	return false
}
