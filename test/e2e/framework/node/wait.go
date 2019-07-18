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
	"fmt"
	"regexp"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/util/system"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	testutils "k8s.io/kubernetes/test/utils"
)

const sleepTime = 20 * time.Second

var requiredPerNodePods = []*regexp.Regexp{
	regexp.MustCompile(".*kube-proxy.*"),
	regexp.MustCompile(".*fluentd-elasticsearch.*"),
	regexp.MustCompile(".*node-problem-detector.*"),
}

// WaitForReadyNodes waits up to timeout for cluster to has desired size and
// there is no not-ready nodes in it. By cluster size we mean number of Nodes
// excluding Master Node.
func WaitForReadyNodes(c clientset.Interface, size int, timeout time.Duration) error {
	_, err := CheckReady(c, size, timeout)
	return err
}

// WaitForTotalHealthy checks whether all registered nodes are ready and all required Pods are running on them.
func WaitForTotalHealthy(c clientset.Interface, timeout time.Duration) error {
	e2elog.Logf("Waiting up to %v for all nodes to be ready", timeout)

	var notReady []v1.Node
	var missingPodsPerNode map[string][]string
	err := wait.PollImmediate(poll, timeout, func() (bool, error) {
		notReady = nil
		// It should be OK to list unschedulable Nodes here.
		nodes, err := c.CoreV1().Nodes().List(metav1.ListOptions{ResourceVersion: "0"})
		if err != nil {
			if testutils.IsRetryableAPIError(err) {
				return false, nil
			}
			return false, err
		}
		for _, node := range nodes.Items {
			if !IsConditionSetAsExpected(&node, v1.NodeReady, true) {
				notReady = append(notReady, node)
			}
		}
		pods, err := c.CoreV1().Pods(metav1.NamespaceAll).List(metav1.ListOptions{ResourceVersion: "0"})
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
			if !system.IsMasterNode(node.Name) {
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

	if err != nil && err != wait.ErrWaitTimeout {
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
func WaitConditionToBe(c clientset.Interface, name string, conditionType v1.NodeConditionType, wantTrue bool, timeout time.Duration) bool {
	e2elog.Logf("Waiting up to %v for node %s condition %s to be %t", timeout, name, conditionType, wantTrue)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(poll) {
		node, err := c.CoreV1().Nodes().Get(name, metav1.GetOptions{})
		if err != nil {
			e2elog.Logf("Couldn't get node %s", name)
			continue
		}

		if IsConditionSetAsExpected(node, conditionType, wantTrue) {
			return true
		}
	}
	e2elog.Logf("Node %s didn't reach desired %s condition status (%t) within %v", name, conditionType, wantTrue, timeout)
	return false
}

// WaitForNodeToBeNotReady returns whether node name is not ready (i.e. the
// readiness condition is anything but ready, e.g false or unknown) within
// timeout.
func WaitForNodeToBeNotReady(c clientset.Interface, name string, timeout time.Duration) bool {
	return WaitConditionToBe(c, name, v1.NodeReady, false, timeout)
}

// WaitForNodeToBeReady returns whether node name is ready within timeout.
func WaitForNodeToBeReady(c clientset.Interface, name string, timeout time.Duration) bool {
	return WaitConditionToBe(c, name, v1.NodeReady, true, timeout)
}

// CheckReady waits up to timeout for cluster to has desired size and
// there is no not-ready nodes in it. By cluster size we mean number of Nodes
// excluding Master Node.
func CheckReady(c clientset.Interface, size int, timeout time.Duration) ([]v1.Node, error) {
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(sleepTime) {
		nodes, err := waitListSchedulableNodes(c)
		if err != nil {
			e2elog.Logf("Failed to list nodes: %v", err)
			continue
		}
		numNodes := len(nodes.Items)

		// Filter out not-ready nodes.
		Filter(nodes, func(node v1.Node) bool {
			nodeReady := IsConditionSetAsExpected(&node, v1.NodeReady, true)
			networkReady := IsConditionUnset(&node, v1.NodeNetworkUnavailable) || IsConditionSetAsExpected(&node, v1.NodeNetworkUnavailable, false)
			return nodeReady && networkReady
		})
		numReady := len(nodes.Items)

		if numNodes == size && numReady == size {
			e2elog.Logf("Cluster has reached the desired number of ready nodes %d", size)
			return nodes.Items, nil
		}
		e2elog.Logf("Waiting for ready nodes %d, current ready %d, not ready nodes %d", size, numReady, numNodes-numReady)
	}
	return nil, fmt.Errorf("timeout waiting %v for number of ready nodes to be %d", timeout, size)
}

// waitListSchedulableNodes is a wrapper around listing nodes supporting retries.
func waitListSchedulableNodes(c clientset.Interface) (*v1.NodeList, error) {
	var nodes *v1.NodeList
	var err error
	if wait.PollImmediate(poll, singleCallTimeout, func() (bool, error) {
		nodes, err = c.CoreV1().Nodes().List(metav1.ListOptions{FieldSelector: fields.Set{
			"spec.unschedulable": "false",
		}.AsSelector().String()})
		if err != nil {
			if testutils.IsRetryableAPIError(err) {
				return false, nil
			}
			return false, err
		}
		return true, nil
	}) != nil {
		return nodes, err
	}
	return nodes, nil
}

// checkWaitListSchedulableNodes is a wrapper around listing nodes supporting retries.
func checkWaitListSchedulableNodes(c clientset.Interface) (*v1.NodeList, error) {
	nodes, err := waitListSchedulableNodes(c)
	if err != nil {
		return nil, fmt.Errorf("error: %s. Non-retryable failure or timed out while listing nodes for e2e cluster", err)
	}
	return nodes, nil
}
