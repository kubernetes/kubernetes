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
	"net"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	nodectlr "k8s.io/kubernetes/pkg/controller/nodelifecycle"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
	"k8s.io/kubernetes/pkg/util/system"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	testutils "k8s.io/kubernetes/test/utils"
)

const (
	// poll is how often to Poll pods, nodes and claims.
	poll = 2 * time.Second

	// singleCallTimeout is how long to try single API calls (like 'get' or 'list'). Used to prevent
	// transient failures from failing tests.
	// TODO: client should not apply this timeout to Watch calls. Increased from 30s until that is fixed.
	singleCallTimeout = 5 * time.Minute

	// ssh port
	sshPort = "22"

	// timeout for proxy requests.
	proxyTimeout = 2 * time.Minute
)

// PodNode is a pod-node pair indicating which node a given pod is running on
type PodNode struct {
	// Pod represents pod name
	Pod string
	// Node represents node name
	Node string
}

// FirstAddress returns the first address of the given type of each node.
// TODO: Use return type string instead of []string
func FirstAddress(nodelist *v1.NodeList, addrType v1.NodeAddressType) []string {
	hosts := []string{}
	for _, n := range nodelist.Items {
		for _, addr := range n.Status.Addresses {
			if addr.Type == addrType && addr.Address != "" {
				hosts = append(hosts, addr.Address)
				break
			}
		}
	}
	return hosts
}

// TODO: better to change to a easy read name
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
					}
					msg = fmt.Sprintf("Condition %s of node %s is %v, but Node is tainted by NodeController with %v. Failure",
						conditionType, node.Name, cond.Status == v1.ConditionTrue, taints)
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

// IsConditionUnset returns true if conditions of the given node do not have a match to the given conditionType, otherwise false.
func IsConditionUnset(node *v1.Node, conditionType v1.NodeConditionType) bool {
	for _, cond := range node.Status.Conditions {
		if cond.Type == conditionType {
			return false
		}
	}
	return true
}

// Filter filters nodes in NodeList in place, removing nodes that do not
// satisfy the given condition
// TODO: consider merging with pkg/client/cache.NodeLister
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

// getSvcNodePort returns the node port for the given service:port.
func getSvcNodePort(client clientset.Interface, ns, name string, svcPort int) (int, error) {
	svc, err := client.CoreV1().Services(ns).Get(name, metav1.GetOptions{})
	if err != nil {
		return 0, err
	}
	for _, p := range svc.Spec.Ports {
		if p.Port == int32(svcPort) {
			if p.NodePort != 0 {
				return int(p.NodePort), nil
			}
		}
	}
	return 0, fmt.Errorf(
		"No node port found for service %v, port %v", name, svcPort)
}

// GetPortURL returns the url to a nodeport Service.
func GetPortURL(client clientset.Interface, ns, name string, svcPort int) (string, error) {
	nodePort, err := getSvcNodePort(client, ns, name, svcPort)
	if err != nil {
		return "", err
	}
	// This list of nodes must not include the master, which is marked
	// unschedulable, since the master doesn't run kube-proxy. Without
	// kube-proxy NodePorts won't work.
	var nodes *v1.NodeList
	if wait.PollImmediate(poll, singleCallTimeout, func() (bool, error) {
		nodes, err = client.CoreV1().Nodes().List(metav1.ListOptions{FieldSelector: fields.Set{
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
		return "", err
	}
	if len(nodes.Items) == 0 {
		return "", fmt.Errorf("Unable to list nodes in cluster")
	}
	for _, node := range nodes.Items {
		for _, address := range node.Status.Addresses {
			if address.Type == v1.NodeExternalIP {
				if address.Address != "" {
					return fmt.Sprintf("http://%v:%v", address.Address, nodePort), nil
				}
			}
		}
	}
	return "", fmt.Errorf("Failed to find external address for service %v", name)
}

// ProxyRequest performs a get on a node proxy endpoint given the nodename and rest client.
func ProxyRequest(c clientset.Interface, node, endpoint string, port int) (restclient.Result, error) {
	// proxy tends to hang in some cases when Node is not ready. Add an artificial timeout for this call.
	// This will leak a goroutine if proxy hangs. #22165
	var result restclient.Result
	finished := make(chan struct{})
	go func() {
		result = c.CoreV1().RESTClient().Get().
			Resource("nodes").
			SubResource("proxy").
			Name(fmt.Sprintf("%v:%v", node, port)).
			Suffix(endpoint).
			Do()

		finished <- struct{}{}
	}()
	select {
	case <-finished:
		return result, nil
	case <-time.After(proxyTimeout):
		return restclient.Result{}, nil
	}
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
		if address.Type == v1.NodeInternalIP {
			if address.Address != "" {
				host = net.JoinHostPort(address.Address, sshPort)
				break
			}
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
	nodes, err := GetReadySchedulableNodesOrDie(c)
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

// GetReadySchedulableNodesOrDie addresses the common use case of getting nodes you can do work on.
// 1) Needs to be schedulable.
// 2) Needs to be ready.
// If EITHER 1 or 2 is not true, most tests will want to ignore the node entirely.
// TODO: remove references in framework/util.go.
// TODO: remove "OrDie" suffix.
func GetReadySchedulableNodesOrDie(c clientset.Interface) (nodes *v1.NodeList, err error) {
	nodes, err = checkWaitListSchedulableNodes(c)
	if err != nil {
		return nil, fmt.Errorf("listing schedulable nodes error: %s", err)
	}
	// previous tests may have cause failures of some nodes. Let's skip
	// 'Not Ready' nodes, just in case (there is no need to fail the test).
	Filter(nodes, func(node v1.Node) bool {
		return isNodeSchedulable(&node) && isNodeUntainted(&node)
	})
	return nodes, nil
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
		return isNodeSchedulable(&node)
	})
	return nodes, nil
}

// GetMasterAndWorkerNodes will return a list masters and schedulable worker nodes
func GetMasterAndWorkerNodes(c clientset.Interface) (sets.String, *v1.NodeList, error) {
	nodes := &v1.NodeList{}
	masters := sets.NewString()
	all, err := c.CoreV1().Nodes().List(metav1.ListOptions{})
	if err != nil {
		return nil, nil, fmt.Errorf("get nodes error: %s", err)
	}
	for _, n := range all.Items {
		if system.IsMasterNode(n.Name) {
			masters.Insert(n.Name)
		} else if isNodeSchedulable(&n) && isNodeUntainted(&n) {
			nodes.Items = append(nodes.Items, n)
		}
	}
	return masters, nodes, nil
}

// Test whether a fake pod can be scheduled on "node", given its current taints.
// TODO: need to discuss wether to return bool and error type
func isNodeUntainted(node *v1.Node) bool {
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
	nodeInfo := schedulernodeinfo.NewNodeInfo()
	nodeInfo.SetNode(node)
	fit, _, err := predicates.PodToleratesNodeTaints(fakePod, nil, nodeInfo)
	if err != nil {
		e2elog.Failf("Can't test predicates for node %s: %v", node.Name, err)
		return false
	}
	return fit
}

// Node is schedulable if:
// 1) doesn't have "unschedulable" field set
// 2) it's Ready condition is set to true
// 3) doesn't have NetworkUnavailable condition set to true
func isNodeSchedulable(node *v1.Node) bool {
	nodeReady := IsConditionSetAsExpected(node, v1.NodeReady, true)
	networkReady := IsConditionUnset(node, v1.NodeNetworkUnavailable) ||
		IsConditionSetAsExpectedSilent(node, v1.NodeNetworkUnavailable, false)
	return !node.Spec.Unschedulable && nodeReady && networkReady
}

// PodNodePairs return podNode pairs for all pods in a namespace
func PodNodePairs(c clientset.Interface, ns string) ([]PodNode, error) {
	var result []PodNode

	podList, err := c.CoreV1().Pods(ns).List(metav1.ListOptions{})
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
