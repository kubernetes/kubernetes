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
	"net"
	"os"
	"reflect"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	corelisters "k8s.io/client-go/listers/core/v1"
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
	nodeIPs  []net.IP
	logger   klog.Logger
	exitFunc func(exitCode int)
}

// NewNodeManager returns NodeManager after waiting for the node object to exist and have NodeIPs.
func NewNodeManager(ctx context.Context, nodeLister corelisters.NodeLister, nodeName string) *NodeManager {
	return newNodeManager(ctx, nodeLister, nodeName, time.Second, 30*time.Second)
}

// newNodeManager implements NewNodeManager with configurable poll interval and timeout.
func newNodeManager(ctx context.Context, nodeLister corelisters.NodeLister, nodeName string, pollInterval, pollTimeout time.Duration) *NodeManager {
	logger := klog.FromContext(ctx)
	var node *v1.Node
	var err error
	var nodeIPs []net.IP

	ctx, cancel := context.WithTimeout(ctx, pollTimeout)
	defer cancel()
	pollErr := wait.PollUntilContextCancel(ctx, pollInterval, true, func(context.Context) (bool, error) {
		node, err = nodeLister.Get(nodeName)
		if err != nil {
			return false, nil
		}

		nodeIPs, err = utilnode.GetNodeHostIPs(node)
		if err != nil {
			return false, nil
		}
		return true, nil
	})

	if pollErr != nil {
		logger.Error(err, "Failed to retrieve NodeIPs")
	} else {
		logger.Info("Successfully retrieved NodeIPs", "NodeIPs", nodeIPs)
	}
	return &NodeManager{
		nodeIPs:  nodeIPs,
		logger:   klog.FromContext(ctx),
		exitFunc: os.Exit,
	}
}

// NodeIPs returns the NodeIPs polled in NewNodeManager().
func (n *NodeManager) NodeIPs() []net.IP {
	return n.nodeIPs
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
	nodeIPs, err := utilnode.GetNodeHostIPs(node)
	if err != nil {
		n.logger.Error(err, "Failed to retrieve NodeIPs")
		return
	}

	// We exit whenever there is a change in NodeIPs detected initially, and NodeIPs received
	// on node watch event.
	if !reflect.DeepEqual(n.nodeIPs, nodeIPs) {
		n.logger.Error(nil, "NodeIPs changed for the node",
			"node", klog.KObj(node), "newNodeIPs", nodeIPs, "oldNodeIPs", n.nodeIPs)
		klog.Flush()
		n.exitFunc(1)
	}
}

// OnNodeDelete is a handler for Node deletes.
func (n *NodeManager) OnNodeDelete(node *v1.Node) {
	n.logger.Error(nil, "Node is being deleted", "node", klog.KObj(node))
	klog.Flush()
	n.exitFunc(1)
}

// OnNodeSynced is called after the cache is synced and all pre-existing Nodes have been reported
func (n *NodeManager) OnNodeSynced() {}
