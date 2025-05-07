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
	"k8s.io/kubernetes/pkg/proxy/config"
	"k8s.io/kubernetes/pkg/proxy/healthcheck"
	utilnode "k8s.io/kubernetes/pkg/util/node"
)

// NodePodCIDRHandler handles the life cycle of kube-proxy based on the node PodCIDR assigned
// Implements the config.NodeHandler interface
// https://issues.k8s.io/111321
type NodePodCIDRHandler struct {
	mu       sync.Mutex
	podCIDRs []string
	logger   klog.Logger
}

func NewNodePodCIDRHandler(ctx context.Context, podCIDRs []string) *NodePodCIDRHandler {
	return &NodePodCIDRHandler{
		podCIDRs: podCIDRs,
		logger:   klog.FromContext(ctx),
	}
}

var _ config.NodeHandler = &NodePodCIDRHandler{}

// OnNodeAdd is a handler for Node creates.
func (n *NodePodCIDRHandler) OnNodeAdd(node *v1.Node) {
	n.mu.Lock()
	defer n.mu.Unlock()

	podCIDRs := node.Spec.PodCIDRs
	// initialize podCIDRs
	if len(n.podCIDRs) == 0 && len(podCIDRs) > 0 {
		n.logger.Info("Setting current PodCIDRs", "podCIDRs", podCIDRs)
		n.podCIDRs = podCIDRs
		return
	}
	if !reflect.DeepEqual(n.podCIDRs, podCIDRs) {
		n.logger.Error(nil, "Using NodeCIDR LocalDetector mode, current PodCIDRs are different than previous PodCIDRs, restarting",
			"node", klog.KObj(node), "newPodCIDRs", podCIDRs, "oldPodCIDRs", n.podCIDRs)
		klog.FlushAndExit(klog.ExitFlushTimeout, 1)
	}
}

// OnNodeUpdate is a handler for Node updates.
func (n *NodePodCIDRHandler) OnNodeUpdate(_, node *v1.Node) {
	n.mu.Lock()
	defer n.mu.Unlock()
	podCIDRs := node.Spec.PodCIDRs
	// initialize podCIDRs
	if len(n.podCIDRs) == 0 && len(podCIDRs) > 0 {
		n.logger.Info("Setting current PodCIDRs", "podCIDRs", podCIDRs)
		n.podCIDRs = podCIDRs
		return
	}
	if !reflect.DeepEqual(n.podCIDRs, podCIDRs) {
		n.logger.Error(nil, "Using NodeCIDR LocalDetector mode, current PodCIDRs are different than previous PodCIDRs, restarting",
			"node", klog.KObj(node), "newPodCIDRs", podCIDRs, "oldPODCIDRs", n.podCIDRs)
		klog.FlushAndExit(klog.ExitFlushTimeout, 1)
	}
}

// OnNodeDelete is a handler for Node deletes.
func (n *NodePodCIDRHandler) OnNodeDelete(node *v1.Node) {
	n.logger.Error(nil, "Current Node is being deleted", "node", klog.KObj(node))
}

// OnNodeSynced is a handler for Node syncs.
func (n *NodePodCIDRHandler) OnNodeSynced() {}

// NodeEligibleHandler handles the life cycle of the Node's eligibility, as
// determined by the health server for directing load balancer traffic.
type NodeEligibleHandler struct {
	HealthServer *healthcheck.ProxyHealthServer
}

var _ config.NodeHandler = &NodeEligibleHandler{}

// OnNodeAdd is a handler for Node creates.
func (n *NodeEligibleHandler) OnNodeAdd(node *v1.Node) { n.HealthServer.SyncNode(node) }

// OnNodeUpdate is a handler for Node updates.
func (n *NodeEligibleHandler) OnNodeUpdate(_, node *v1.Node) { n.HealthServer.SyncNode(node) }

// OnNodeDelete is a handler for Node deletes.
func (n *NodeEligibleHandler) OnNodeDelete(node *v1.Node) { n.HealthServer.SyncNode(node) }

// OnNodeSynced is a handler for Node syncs.
func (n *NodeEligibleHandler) OnNodeSynced() {}

// NodeManager handles the life cycle of kube-proxy based on the NodeIPs, handles node watch events
// and crashes kube-proxy if there are any changes in NodeIPs.
type NodeManager struct {
	nodeInformer v1informers.NodeInformer
	nodeLister   corelisters.NodeLister
	exitFunc     func(exitCode int)

	nodeIPs []net.IP
}

// NewNodeManager initializes node informer that selects for the given node, waits for cache sync
// and returns NodeManager after waiting some amount of time for the node object to exist
// and have NodeIPs. Note: NewNodeManager doesn't return any error if failed to retrieve NodeIPs.
func NewNodeManager(ctx context.Context, client clientset.Interface, resyncInterval time.Duration, nodeName string) (*NodeManager, error) {
	return newNodeManager(ctx, client, resyncInterval, nodeName, os.Exit, time.Second, 30*time.Second)
}

// newNodeManager implements NewNodeManager with configurable exit function, poll interval and timeout.
func newNodeManager(ctx context.Context, client clientset.Interface, resyncInterval time.Duration, nodeName string, exitFunc func(int), pollInterval, pollTimeout time.Duration) (*NodeManager, error) {
	// make an informer that selects for the given node
	thisNodeInformerFactory := informers.NewSharedInformerFactoryWithOptions(client, resyncInterval,
		informers.WithTweakListOptions(func(options *metav1.ListOptions) {
			options.FieldSelector = fields.OneTermEqualSelector("metadata.name", nodeName).String()
		}))
	nodeInformer := thisNodeInformerFactory.Core().V1().Nodes()
	nodeLister := nodeInformer.Lister()

	// initialize the informer and wait for cache sync
	thisNodeInformerFactory.Start(wait.NeverStop)
	if !cache.WaitForNamedCacheSync("node informer cache", ctx.Done(), nodeInformer.Informer().HasSynced) {
		return nil, fmt.Errorf("can not sync node informer")
	}

	node, nodeIPs := getNodeInfo(nodeLister, nodeName)

	if len(nodeIPs) == 0 {
		// wait for the node object to (hopefully) exist and have NodeIPs.
		ctx, cancel := context.WithTimeout(ctx, pollTimeout)
		defer cancel()
		_ = wait.PollUntilContextCancel(ctx, pollInterval, true, func(context.Context) (bool, error) {
			node, nodeIPs = getNodeInfo(nodeLister, nodeName)
			return len(nodeIPs) != 0, nil
		})
	}

	// For backward-compatibility, we keep going even if we didn't find a node or it
	// didn't have IPs.
	if node == nil {
		klog.FromContext(ctx).Error(nil, "Timed out waiting for node %q to exist", nodeName)
	} else if len(nodeIPs) == 0 {
		klog.FromContext(ctx).Error(nil, "Timed out waiting for node %q to be assigned IPs", nodeName)
	}

	return &NodeManager{
		nodeInformer: nodeInformer,
		nodeLister:   nodeLister,
		exitFunc:     exitFunc,

		nodeIPs: nodeIPs,
	}, nil
}

func getNodeInfo(nodeLister corelisters.NodeLister, nodeName string) (*v1.Node, []net.IP) {
	node, _ := nodeLister.Get(nodeName)
	if node == nil {
		return nil, nil
	}
	nodeIPs, _ := utilnode.GetNodeHostIPs(node)
	return node, nodeIPs
}

// NodeIPs returns the NodeIPs polled in NewNodeManager(). (This may be empty if
// NewNodeManager timed out without getting any IPs.)
func (n *NodeManager) NodeIPs() []net.IP {
	return n.nodeIPs
}

// NodeInformer returns the NodeInformer.
func (n *NodeManager) NodeInformer() v1informers.NodeInformer {
	return n.nodeInformer
}

// OnNodeAdd is a handler for Node creates.
func (n *NodeManager) OnNodeAdd(node *v1.Node) {
	n.onNodeChange(node)
}

// OnNodeUpdate is a handler for Node updates.
func (n *NodeManager) OnNodeUpdate(_, node *v1.Node) {
	n.onNodeChange(node)
}

// onNodeChange functions helps to implement OnNodeAdd and OnNodeUpdate.
func (n *NodeManager) onNodeChange(node *v1.Node) {
	nodeIPs, _ := utilnode.GetNodeHostIPs(node)

	// We exit whenever there is a change in NodeIPs detected initially, and NodeIPs received
	// on node watch event.
	if !reflect.DeepEqual(n.nodeIPs, nodeIPs) {
		klog.InfoS("NodeIPs changed for the node",
			"node", klog.KObj(node), "newNodeIPs", nodeIPs, "oldNodeIPs", n.nodeIPs)
		klog.Flush()
		n.exitFunc(1)
	}
}

// OnNodeDelete is a handler for Node deletes.
func (n *NodeManager) OnNodeDelete(node *v1.Node) {
	klog.InfoS("Node is being deleted", "node", klog.KObj(node))
	klog.Flush()
	n.exitFunc(1)
}

// OnNodeSynced is called after the cache is synced and all pre-existing Nodes have been reported
func (n *NodeManager) OnNodeSynced() {}
