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
	"net"
	"strings"
	"time"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/controller"
	schedfwk "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	"k8s.io/kubernetes/test/e2e/system"
)

const (
	// poll is how often to Poll pods, nodes and claims.
	poll = 2 * time.Second

	// singleCallTimeout is how long to try single API calls (like 'get' or 'list'). Used to prevent
	// transient failures from failing tests.
	singleCallTimeout = 5 * time.Minute

	// ssh port
	sshPort = "22"
)

var (
	// unreachableTaintTemplate is the taint for when a node becomes unreachable.
	// Copied from pkg/controller/nodelifecycle to avoid pulling extra dependencies
	unreachableTaintTemplate = &v1.Taint{
		Key:    v1.TaintNodeUnreachable,
		Effect: v1.TaintEffectNoExecute,
	}

	// notReadyTaintTemplate is the taint for when a node is not ready for executing pods.
	// Copied from pkg/controller/nodelifecycle to avoid pulling extra dependencies
	notReadyTaintTemplate = &v1.Taint{
		Key:    v1.TaintNodeNotReady,
		Effect: v1.TaintEffectNoExecute,
	}
)

// PodNode is a pod-node pair indicating which node a given pod is running on
type PodNode struct {
	// Pod represents pod name
	Pod string
	// Node represents node name
	Node string
}

// FirstAddress returns the first address of the given type of each node.
func FirstAddress(nodelist *v1.NodeList, addrType v1.NodeAddressType) string {
	for _, n := range nodelist.Items {
		for _, addr := range n.Status.Addresses {
			if addr.Type == addrType && addr.Address != "" {
				return addr.Address
			}
		}
	}
	return ""
}

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
					if taint.MatchTaint(unreachableTaintTemplate) || taint.MatchTaint(notReadyTaintTemplate) {
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
						e2elog.Logf(msg)
					}
					return false
				}
				// TODO: check if the Node is tainted once we enable NC notReady/unreachable taints by default
				if cond.Status != v1.ConditionTrue {
					return true
				}
				if !silent {
					e2elog.Logf("Condition %s of node %s is %v instead of %t. Reason: %v, message: %v",
						conditionType, node.Name, cond.Status == v1.ConditionTrue, wantTrue, cond.Reason, cond.Message)
				}
				return false
			}
			if (wantTrue && (cond.Status == v1.ConditionTrue)) || (!wantTrue && (cond.Status != v1.ConditionTrue)) {
				return true
			}
			if !silent {
				e2elog.Logf("Condition %s of node %s is %v instead of %t. Reason: %v, message: %v",
					conditionType, node.Name, cond.Status == v1.ConditionTrue, wantTrue, cond.Reason, cond.Message)
			}
			return false
		}

	}
	if !silent {
		e2elog.Logf("Couldn't find condition %v on node %v", conditionType, node.Name)
	}
	return false
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

// TotalRegistered returns number of registered Nodes excluding Master Node.
func TotalRegistered(c clientset.Interface) (int, error) {
	nodes, err := waitListSchedulableNodes(c)
	if err != nil {
		e2elog.Logf("Failed to list nodes: %v", err)
		return 0, err
	}
	return len(nodes.Items), nil
}

// TotalReady returns number of ready Nodes excluding Master Node.
func TotalReady(c clientset.Interface) (int, error) {
	nodes, err := waitListSchedulableNodes(c)
	if err != nil {
		e2elog.Logf("Failed to list nodes: %v", err)
		return 0, err
	}

	// Filter out not-ready nodes.
	Filter(nodes, func(node v1.Node) bool {
		return IsConditionSetAsExpected(&node, v1.NodeReady, true)
	})
	return len(nodes.Items), nil
}

// GetExternalIP returns node external IP concatenated with port 22 for ssh
// e.g. 1.2.3.4:22
func GetExternalIP(node *v1.Node) (string, error) {
	e2elog.Logf("Getting external IP address for %s", node.Name)
	host := ""
	for _, a := range node.Status.Addresses {
		if a.Type == v1.NodeExternalIP && a.Address != "" {
			host = net.JoinHostPort(a.Address, sshPort)
			break
		}
	}
	if host == "" {
		return "", fmt.Errorf("Couldn't get the external IP of host %s with addresses %v", node.Name, node.Status.Addresses)
	}
	return host, nil
}

// GetInternalIP returns node internal IP
func GetInternalIP(node *v1.Node) (string, error) {
	host := ""
	for _, address := range node.Status.Addresses {
		if address.Type == v1.NodeInternalIP && address.Address != "" {
			host = net.JoinHostPort(address.Address, sshPort)
			break
		}
	}
	if host == "" {
		return "", fmt.Errorf("Couldn't get the internal IP of host %s with addresses %v", node.Name, node.Status.Addresses)
	}
	return host, nil
}

// GetAddresses returns a list of addresses of the given addressType for the given node
func GetAddresses(node *v1.Node, addressType v1.NodeAddressType) (ips []string) {
	for j := range node.Status.Addresses {
		nodeAddress := &node.Status.Addresses[j]
		if nodeAddress.Type == addressType && nodeAddress.Address != "" {
			ips = append(ips, nodeAddress.Address)
		}
	}
	return
}

// CollectAddresses returns a list of addresses of the given addressType for the given list of nodes
func CollectAddresses(nodes *v1.NodeList, addressType v1.NodeAddressType) []string {
	ips := []string{}
	for i := range nodes.Items {
		ips = append(ips, GetAddresses(&nodes.Items[i], addressType)...)
	}
	return ips
}

// PickIP picks one public node IP
func PickIP(c clientset.Interface) (string, error) {
	publicIps, err := GetPublicIps(c)
	if err != nil {
		return "", fmt.Errorf("get node public IPs error: %s", err)
	}
	if len(publicIps) == 0 {
		return "", fmt.Errorf("got unexpected number (%d) of public IPs", len(publicIps))
	}
	ip := publicIps[0]
	return ip, nil
}

// GetPublicIps returns a public IP list of nodes.
func GetPublicIps(c clientset.Interface) ([]string, error) {
	nodes, err := GetReadySchedulableNodes(c)
	if err != nil {
		return nil, fmt.Errorf("get schedulable and ready nodes error: %s", err)
	}
	ips := CollectAddresses(nodes, v1.NodeExternalIP)
	if len(ips) == 0 {
		// If ExternalIP isn't set, assume the test programs can reach the InternalIP
		ips = CollectAddresses(nodes, v1.NodeInternalIP)
	}
	return ips, nil
}

// GetReadySchedulableNodes addresses the common use case of getting nodes you can do work on.
// 1) Needs to be schedulable.
// 2) Needs to be ready.
// If EITHER 1 or 2 is not true, most tests will want to ignore the node entirely.
// If there are no nodes that are both ready and schedulable, this will return an error.
func GetReadySchedulableNodes(c clientset.Interface) (nodes *v1.NodeList, err error) {
	nodes, err = checkWaitListSchedulableNodes(c)
	if err != nil {
		return nil, fmt.Errorf("listing schedulable nodes error: %s", err)
	}
	Filter(nodes, func(node v1.Node) bool {
		return IsNodeSchedulable(&node) && isNodeUntainted(&node)
	})
	if len(nodes.Items) == 0 {
		return nil, fmt.Errorf("there are currently no ready, schedulable nodes in the cluster")
	}
	return nodes, nil
}

// GetBoundedReadySchedulableNodes is like GetReadySchedulableNodes except that it returns
// at most maxNodes nodes. Use this to keep your test case from blowing up when run on a
// large cluster.
func GetBoundedReadySchedulableNodes(c clientset.Interface, maxNodes int) (nodes *v1.NodeList, err error) {
	nodes, err = GetReadySchedulableNodes(c)
	if err != nil {
		return nil, err
	}
	if len(nodes.Items) > maxNodes {
		shuffled := make([]v1.Node, maxNodes)
		perm := rand.Perm(len(nodes.Items))
		for i, j := range perm {
			if j < len(shuffled) {
				shuffled[j] = nodes.Items[i]
			}
		}
		nodes.Items = shuffled
	}
	return nodes, nil
}

// GetRandomReadySchedulableNode gets a single randomly-selected node which is available for
// running pods on. If there are no available nodes it will return an error.
func GetRandomReadySchedulableNode(c clientset.Interface) (*v1.Node, error) {
	nodes, err := GetReadySchedulableNodes(c)
	if err != nil {
		return nil, err
	}
	return &nodes.Items[rand.Intn(len(nodes.Items))], nil
}

// GetReadyNodesIncludingTainted returns all ready nodes, even those which are tainted.
// There are cases when we care about tainted nodes
// E.g. in tests related to nodes with gpu we care about nodes despite
// presence of nvidia.com/gpu=present:NoSchedule taint
func GetReadyNodesIncludingTainted(c clientset.Interface) (nodes *v1.NodeList, err error) {
	nodes, err = checkWaitListSchedulableNodes(c)
	if err != nil {
		return nil, fmt.Errorf("listing schedulable nodes error: %s", err)
	}
	Filter(nodes, func(node v1.Node) bool {
		return IsNodeSchedulable(&node)
	})
	return nodes, nil
}

// GetMasterAndWorkerNodes will return a list masters and schedulable worker nodes
func GetMasterAndWorkerNodes(c clientset.Interface) (sets.String, *v1.NodeList, error) {
	nodes := &v1.NodeList{}
	masters := sets.NewString()
	all, err := c.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		return nil, nil, fmt.Errorf("get nodes error: %s", err)
	}
	for _, n := range all.Items {
		if system.DeprecatedMightBeMasterNode(n.Name) {
			masters.Insert(n.Name)
		} else if IsNodeSchedulable(&n) && isNodeUntainted(&n) {
			nodes.Items = append(nodes.Items, n)
		}
	}
	return masters, nodes, nil
}

// isNodeUntainted tests whether a fake pod can be scheduled on "node", given its current taints.
// TODO: need to discuss wether to return bool and error type
func isNodeUntainted(node *v1.Node) bool {
	return isNodeUntaintedWithNonblocking(node, "")
}

// isNodeUntaintedWithNonblocking tests whether a fake pod can be scheduled on "node"
// but allows for taints in the list of non-blocking taints.
func isNodeUntaintedWithNonblocking(node *v1.Node, nonblockingTaints string) bool {
	fakePod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "fake-not-scheduled",
			Namespace: "fake-not-scheduled",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "fake-not-scheduled",
					Image: "fake-not-scheduled",
				},
			},
		},
	}

	nodeInfo := schedfwk.NewNodeInfo()

	// Simple lookup for nonblocking taints based on comma-delimited list.
	nonblockingTaintsMap := map[string]struct{}{}
	for _, t := range strings.Split(nonblockingTaints, ",") {
		if strings.TrimSpace(t) != "" {
			nonblockingTaintsMap[strings.TrimSpace(t)] = struct{}{}
		}
	}

	if len(nonblockingTaintsMap) > 0 {
		nodeCopy := node.DeepCopy()
		nodeCopy.Spec.Taints = []v1.Taint{}
		for _, v := range node.Spec.Taints {
			if _, isNonblockingTaint := nonblockingTaintsMap[v.Key]; !isNonblockingTaint {
				nodeCopy.Spec.Taints = append(nodeCopy.Spec.Taints, v)
			}
		}
		nodeInfo.SetNode(nodeCopy)
	} else {
		nodeInfo.SetNode(node)
	}

	taints, err := nodeInfo.Taints()
	if err != nil {
		e2elog.Failf("Can't test predicates for node %s: %v", node.Name, err)
		return false
	}

	return toleratesTaintsWithNoScheduleNoExecuteEffects(taints, fakePod.Spec.Tolerations)
}

func toleratesTaintsWithNoScheduleNoExecuteEffects(taints []v1.Taint, tolerations []v1.Toleration) bool {
	filteredTaints := []v1.Taint{}
	for _, taint := range taints {
		if taint.Effect == v1.TaintEffectNoExecute || taint.Effect == v1.TaintEffectNoSchedule {
			filteredTaints = append(filteredTaints, taint)
		}
	}

	toleratesTaint := func(taint v1.Taint) bool {
		for _, toleration := range tolerations {
			if toleration.ToleratesTaint(&taint) {
				return true
			}
		}

		return false
	}

	for _, taint := range filteredTaints {
		if !toleratesTaint(taint) {
			return false
		}
	}

	return true
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

// hasNonblockingTaint returns true if the node contains at least
// one taint with a key matching the regexp.
func hasNonblockingTaint(node *v1.Node, nonblockingTaints string) bool {
	if node == nil {
		return false
	}

	// Simple lookup for nonblocking taints based on comma-delimited list.
	nonblockingTaintsMap := map[string]struct{}{}
	for _, t := range strings.Split(nonblockingTaints, ",") {
		if strings.TrimSpace(t) != "" {
			nonblockingTaintsMap[strings.TrimSpace(t)] = struct{}{}
		}
	}

	for _, taint := range node.Spec.Taints {
		if _, hasNonblockingTaint := nonblockingTaintsMap[taint.Key]; hasNonblockingTaint {
			return true
		}
	}

	return false
}

// PodNodePairs return podNode pairs for all pods in a namespace
func PodNodePairs(c clientset.Interface, ns string) ([]PodNode, error) {
	var result []PodNode

	podList, err := c.CoreV1().Pods(ns).List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		return result, err
	}

	for _, pod := range podList.Items {
		result = append(result, PodNode{
			Pod:  pod.Name,
			Node: pod.Spec.NodeName,
		})
	}

	return result, nil
}

// GetClusterZones returns the values of zone label collected from all nodes.
func GetClusterZones(c clientset.Interface) (sets.String, error) {
	nodes, err := c.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		return nil, fmt.Errorf("Error getting nodes while attempting to list cluster zones: %v", err)
	}

	// collect values of zone label from all nodes
	zones := sets.NewString()
	for _, node := range nodes.Items {
		if zone, found := node.Labels[v1.LabelZoneFailureDomain]; found {
			zones.Insert(zone)
		}

		if zone, found := node.Labels[v1.LabelZoneFailureDomainStable]; found {
			zones.Insert(zone)
		}
	}
	return zones, nil
}

// CreatePodsPerNodeForSimpleApp creates pods w/ labels.  Useful for tests which make a bunch of pods w/o any networking.
func CreatePodsPerNodeForSimpleApp(c clientset.Interface, namespace, appName string, podSpec func(n v1.Node) v1.PodSpec, maxCount int) map[string]string {
	nodes, err := GetBoundedReadySchedulableNodes(c, maxCount)
	// TODO use wrapper methods in expect.go after removing core e2e dependency on node
	gomega.ExpectWithOffset(2, err).NotTo(gomega.HaveOccurred())
	podLabels := map[string]string{
		"app": appName + "-pod",
	}
	for i, node := range nodes.Items {
		e2elog.Logf("%v/%v : Creating container with label app=%v-pod", i, maxCount, appName)
		_, err := c.CoreV1().Pods(namespace).Create(context.TODO(), &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:   fmt.Sprintf(appName+"-pod-%v", i),
				Labels: podLabels,
			},
			Spec: podSpec(node),
		}, metav1.CreateOptions{})
		// TODO use wrapper methods in expect.go after removing core e2e dependency on node
		gomega.ExpectWithOffset(2, err).NotTo(gomega.HaveOccurred())
	}
	return podLabels
}

// RemoveTaintOffNode removes the given taint from the given node.
func RemoveTaintOffNode(c clientset.Interface, nodeName string, taint v1.Taint) {
	err := controller.RemoveTaintOffNode(c, nodeName, nil, &taint)

	// TODO use wrapper methods in expect.go after removing core e2e dependency on node
	gomega.ExpectWithOffset(2, err).NotTo(gomega.HaveOccurred())
	verifyThatTaintIsGone(c, nodeName, &taint)
}

func verifyThatTaintIsGone(c clientset.Interface, nodeName string, taint *v1.Taint) {
	ginkgo.By("verifying the node doesn't have the taint " + taint.ToString())
	nodeUpdated, err := c.CoreV1().Nodes().Get(context.TODO(), nodeName, metav1.GetOptions{})

	// TODO use wrapper methods in expect.go after removing core e2e dependency on node
	gomega.ExpectWithOffset(2, err).NotTo(gomega.HaveOccurred())
	if taintExists(nodeUpdated.Spec.Taints, taint) {
		e2elog.Failf("Failed removing taint " + taint.ToString() + " of the node " + nodeName)
	}
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
