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
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/system"
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

// TotalReady returns number of ready Nodes excluding Master Node.
func TotalReady(c clientset.Interface) (int, error) {
	nodes, err := waitListSchedulableNodes(c)
	if err != nil {
		framework.Logf("Failed to list nodes: %v", err)
		return 0, err
	}

	// Filter out not-ready nodes.
	framework.Filter(nodes, func(node v1.Node) bool {
		return framework.IsConditionSetAsExpected(&node, v1.NodeReady, true)
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
					host := net.JoinHostPort(address.Address, fmt.Sprint(nodePort))
					return fmt.Sprintf("http://%s", host), nil
				}
			}
		}
	}
	return "", fmt.Errorf("Failed to find external address for service %v", name)
}

// GetExternalIP returns node external IP concatenated with port 22 for ssh
// e.g. 1.2.3.4:22
func GetExternalIP(node *v1.Node) (string, error) {
	framework.Logf("Getting external IP address for %s", node.Name)
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
	nodes, err := framework.GetReadySchedulableNodes(c)
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

// GetRandomReadySchedulableNode gets a single randomly-selected node which is available for
// running pods on. If there are no available nodes it will return an error.
func GetRandomReadySchedulableNode(c clientset.Interface) (*v1.Node, error) {
	nodes, err := framework.GetReadySchedulableNodes(c)
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
	framework.Filter(nodes, func(node v1.Node) bool {
		return framework.IsNodeSchedulable(&node)
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
		if system.DeprecatedMightBeMasterNode(n.Name) {
			masters.Insert(n.Name)
		} else if framework.IsNodeSchedulable(&n) && framework.IsNodeUntainted(&n) {
			nodes.Items = append(nodes.Items, n)
		}
	}
	return masters, nodes, nil
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
