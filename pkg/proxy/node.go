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
	"reflect"
	"sync"

	v1 "k8s.io/api/core/v1"
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

// NodePodCIDRHandler handles the life cycle of kube-proxy based on the node PodCIDR assigned
// Implements the config.NodeHandler interface
// https://issues.k8s.io/111321
type NodeIPsHandler struct {
	mu      sync.Mutex
	nodeIPs []net.IP
	logger  klog.Logger
}

func NewNodeIPsHandler(ctx context.Context, nodeIPs []net.IP) *NodeIPsHandler {
	return &NodeIPsHandler{
		nodeIPs: nodeIPs,
		logger:  klog.FromContext(ctx),
	}
}

var _ config.NodeHandler = &NodeIPsHandler{}

// OnNodeAdd is a handler for Node creates.
func (n *NodeIPsHandler) OnNodeAdd(node *v1.Node) {
	n.mu.Lock()
	defer n.mu.Unlock()

	nodeIPs, err := utilnode.GetNodeHostIPs(node)
	if err != nil {
		n.logger.Error(err, "Failed to retrieve node IPs")
		return
	}

	// initialize podCIDRs
	if len(n.nodeIPs) == 0 && len(nodeIPs) > 0 {
		n.logger.Info("Setting current NodeIPs", "NodeIPs", nodeIPs)
		n.nodeIPs = nodeIPs
		return
	}
	if !reflect.DeepEqual(n.nodeIPs, nodeIPs) {
		n.logger.Error(nil, "current NodeIPs are different than previous NodeIPs, restarting",
			"node", klog.KObj(node), "newNodeIPs", nodeIPs, "oldNodeIPs", n.nodeIPs)
		klog.FlushAndExit(klog.ExitFlushTimeout, 1)
	}
}

// OnNodeUpdate is a handler for Node updates.
func (n *NodeIPsHandler) OnNodeUpdate(_, node *v1.Node) {
	n.mu.Lock()
	defer n.mu.Unlock()

	nodeIPs, err := utilnode.GetNodeHostIPs(node)
	if err != nil {
		n.logger.Error(err, "Failed to retrieve node IPs")
		return
	}
	// initialize podCIDRs
	if len(n.nodeIPs) == 0 && len(nodeIPs) > 0 {
		n.logger.Info("Setting current NodeIPs", "NodeIPs", nodeIPs)
		n.nodeIPs = nodeIPs
		return
	}
	if !reflect.DeepEqual(n.nodeIPs, nodeIPs) {
		n.logger.Error(nil, "current NodeIPs are different than previous NodeIPs, restarting",
			"node", klog.KObj(node), "newNodeIPs", nodeIPs, "oldNodeIPs", n.nodeIPs)
		klog.FlushAndExit(klog.ExitFlushTimeout, 1)
	}
}

// OnNodeDelete is a handler for Node deletes.
func (n *NodeIPsHandler) OnNodeDelete(node *v1.Node) {
	n.logger.Error(nil, "Current Node is being deleted", "node", klog.KObj(node))
}

// OnNodeSynced is a handler for Node syncs.
func (n *NodeIPsHandler) OnNodeSynced() {}

// NodeEligibleHandler handles the life cycle of the Node's eligibility, as
// determined by the health server for directing load balancer traffic.
type NodeEligibleHandler struct {
	HealthServer *healthcheck.ProxierHealthServer
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
