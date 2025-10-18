/*
Copyright 2022 The Kubernetes Authors.

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

package proxy

import (
	"context"
	"fmt"
	"net"
	"os"
	"reflect"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	v1informers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
	utilnode "k8s.io/kubernetes/pkg/util/node"
)

// NodeManager handles the life cycle of kube-proxy based on the NodeIPs and PodCIDRs handles
// node watch events and crashes kube-proxy if there are any changes in NodeIPs or PodCIDRs.
// Note: It only crashes on change on PodCIDR when watchPodCIDRs is set to true.
type NodeManager struct {
	nodeInformer  v1informers.NodeInformer
	nodeLister    corelisters.NodeLister
	exitFunc      func(exitCode int)
	watchPodCIDRs bool

	// These are constant after construct time
	nodeIPs  []net.IP
	podCIDRs []string

	mu   sync.Mutex
	node *v1.Node
}

// NewNodeManager initializes node informer that selects for the given node, waits for cache sync
// and returns NodeManager after waiting some amount of time for the node object to exist
// and have NodeIPs (and PodCIDRs if watchPodCIDRs is true). Note: for backward compatibility,
// NewNodeManager doesn't return any error if it failed to retrieve NodeIPs and watchPodCIDRs
// is false.
func NewNodeManager(ctx context.Context, client clientset.Interface,
	resyncInterval time.Duration, nodeName string, watchPodCIDRs bool,
) (*NodeManager, error) {
	return newNodeManager(ctx, client, resyncInterval, nodeName, watchPodCIDRs, os.Exit, time.Second, 30*time.Second, 5*time.Minute)
}

// newNodeManager implements NewNodeManager with configurable exit function, poll interval and timeouts.
func newNodeManager(ctx context.Context, client clientset.Interface, resyncInterval time.Duration,
	nodeName string, watchPodCIDRs bool, exitFunc func(int),
	pollInterval, nodeIPsTimeout, podCIDRsTimeout time.Duration,
) (*NodeManager, error) {
	// make an informer that selects for the given node
	thisNodeInformerFactory := informers.NewSharedInformerFactoryWithOptions(client, resyncInterval,
		informers.WithTweakListOptions(func(options *metav1.ListOptions) {
			options.FieldSelector = fields.OneTermEqualSelector("metadata.name", nodeName).String()
		}))
	nodeInformer := thisNodeInformerFactory.Core().V1().Nodes()
	nodeLister := nodeInformer.Lister()

	// initialize the informer and wait for cache sync
	thisNodeInformerFactory.Start(wait.NeverStop)
	if !cache.WaitForNamedCacheSyncWithContext(ctx, nodeInformer.Informer().HasSynced) {
		return nil, fmt.Errorf("can not sync node informer")
	}

	node, nodeIPs, podCIDRs := getNodeInfo(nodeLister, nodeName)

	if len(nodeIPs) == 0 {
		// wait for the node object to exist and have NodeIPs.
		ctx, cancel := context.WithTimeout(ctx, nodeIPsTimeout)
		defer cancel()
		_ = wait.PollUntilContextCancel(ctx, pollInterval, false, func(context.Context) (bool, error) {
			node, nodeIPs, podCIDRs = getNodeInfo(nodeLister, nodeName)
			return len(nodeIPs) != 0, nil
		})
	}

	if watchPodCIDRs && len(podCIDRs) == 0 {
		// wait some additional time for the PodCIDRs.
		ctx, cancel := context.WithTimeout(ctx, podCIDRsTimeout)
		defer cancel()
		_ = wait.PollUntilContextCancel(ctx, pollInterval, false, func(context.Context) (bool, error) {
			node, nodeIPs, podCIDRs = getNodeInfo(nodeLister, nodeName)
			return len(podCIDRs) != 0, nil
		})

		if len(podCIDRs) == 0 {
			if node == nil {
				return nil, fmt.Errorf("timeout waiting for node %q to exist", nodeName)
			} else {
				return nil, fmt.Errorf("timeout waiting for PodCIDR allocation on node %q", nodeName)
			}
		}
	}

	// For backward-compatibility, we keep going even if we didn't find a node (in
	// non-watchPodCIDRs mode) or it didn't have IPs.
	if node == nil {
		klog.FromContext(ctx).Error(nil, "Timed out waiting for node %q to exist", nodeName)
	} else if len(nodeIPs) == 0 {
		klog.FromContext(ctx).Error(nil, "Timed out waiting for node %q to be assigned IPs", nodeName)
	}

	return &NodeManager{
		nodeInformer:  nodeInformer,
		nodeLister:    nodeLister,
		exitFunc:      exitFunc,
		watchPodCIDRs: watchPodCIDRs,

		node:     node,
		nodeIPs:  nodeIPs,
		podCIDRs: podCIDRs,
	}, nil
}

func getNodeInfo(nodeLister corelisters.NodeLister, nodeName string) (*v1.Node, []net.IP, []string) {
	node, _ := nodeLister.Get(nodeName)
	if node == nil {
		return nil, nil, nil
	}
	nodeIPs, _ := utilnode.GetNodeHostIPs(node)
	return node, nodeIPs, node.Spec.PodCIDRs
}

// NodeIPs returns the NodeIPs polled in NewNodeManager(). (This may be empty if
// NewNodeManager timed out without getting any IPs.)
func (n *NodeManager) NodeIPs() []net.IP {
	return n.nodeIPs
}

// PodCIDRs returns the PodCIDRs polled in NewNodeManager().
func (n *NodeManager) PodCIDRs() []string {
	return n.podCIDRs
}

// Node returns a copy of the latest node object, or nil if the Node has not yet been seen.
func (n *NodeManager) Node() *v1.Node {
	n.mu.Lock()
	defer n.mu.Unlock()

	if n.node == nil {
		return nil
	}
	return n.node.DeepCopy()
}

// NodeInformer returns the NodeInformer.
func (n *NodeManager) NodeInformer() v1informers.NodeInformer {
	return n.nodeInformer
}

// OnNodeChange is a handler for Node creation and update.
func (n *NodeManager) OnNodeChange(node *v1.Node) {
	// update the node object
	n.mu.Lock()
	n.node = node
	n.mu.Unlock()

	// We exit whenever there is a change in PodCIDRs detected initially, and PodCIDRs received
	// on node watch event if the node manager is configured with watchPodCIDRs.
	if n.watchPodCIDRs {
		if !reflect.DeepEqual(n.podCIDRs, node.Spec.PodCIDRs) {
			klog.InfoS("PodCIDRs changed for the node",
				"node", klog.KObj(node), "newPodCIDRs", node.Spec.PodCIDRs, "oldPodCIDRs", n.podCIDRs)
			klog.Flush()
			n.exitFunc(1)
		}
	}

	nodeIPs, _ := utilnode.GetNodeHostIPs(node)

	// We exit whenever there is a change in NodeIPs detected initially, and NodeIPs received
	// on node watch event.
	if !reflect.DeepEqual(n.nodeIPs, nodeIPs) {
		klog.InfoS("NodeIPs changed for the node",
			"node", klog.KObj(node), "newNodeIPs", nodeIPs, "oldNodeIPs", n.nodeIPs)
		// FIXME: exit
		// klog.Flush()
		// n.exitFunc(1)
	}
}

// OnNodeDelete is a handler for Node deletes.
func (n *NodeManager) OnNodeDelete(node *v1.Node) {
	klog.InfoS("Node is being deleted", "node", klog.KObj(node))
	// FIXME: exit
	// klog.Flush()
	// n.exitFunc(1)
}

// OnNodeSynced is called after the cache is synced and all pre-existing Nodes have been reported
func (n *NodeManager) OnNodeSynced() {}
