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

	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/proxy/config"
	"k8s.io/kubernetes/pkg/proxy/healthcheck"
	utilnode "k8s.io/kubernetes/pkg/util/node"
)

// NodeManager handles the life cycle of kube-proxy based on the NodeIPs and PodCIDRs.
// Implements the config.NodeHandler interface
// (ref: https://issues.k8s.io/111321)
type NodeManager struct {
	nodeIPs           []net.IP
	podCIDRs          []string
	localModeNodeCIDR bool
	healthServer      *healthcheck.ProxyHealthServer
	logger            klog.Logger
}

func NewNodeManager(ctx context.Context, nodeIPs []net.IP, podCIDRs []string, localModeNodeCIDR bool, healthServer *healthcheck.ProxyHealthServer) *NodeManager {
	return &NodeManager{
		nodeIPs:           nodeIPs,
		podCIDRs:          podCIDRs,
		localModeNodeCIDR: localModeNodeCIDR,
		healthServer:      healthServer,
		logger:            klog.FromContext(ctx),
	}
}

var _ config.NodeHandler = &NodeManager{}

// OnNodeUpsert is a handler for Node upserts.
func (n *NodeManager) OnNodeUpsert(node *v1.Node) {
	n.healthServer.SyncNode(node)
	if n.localModeNodeCIDR {
		podCIDRs := node.Spec.PodCIDRs
		// We exit whenever there is a change in PoDCIDRs detected initially, and PoDCIDRs received
		// on Node watch event. Note that initial PoDCIDRs can never be nil if the LocalModeNodeCIRD
		// is set, as we exit in platformSetup() if failed to wait for PodCIDR.
		if !reflect.DeepEqual(n.podCIDRs, podCIDRs) {
			n.logger.Error(nil, "Using NodeCIDR LocalDetector mode, PodCIDRs changed for the node, restarting",
				"node", klog.KObj(node), "newPodCIDRs", podCIDRs, "oldPodCIDRs", n.podCIDRs)
			klog.FlushAndExit(klog.ExitFlushTimeout, 1)
		}
	}

	nodeIPs, err := utilnode.GetNodeHostIPs(node)
	if err != nil {
		n.logger.Error(err, "Failed to retrieve node IPs")
		return
	}

	// We exit whenever there is a change in NodeIPs detected initially, and NodeIPs received
	// on Node watch event. This includes the case when we fail to detect NodeIPs initially in
	// getNodeIPs() and proceed with empty list of NodeIPs.
	if !reflect.DeepEqual(n.nodeIPs, nodeIPs) {
		n.logger.Error(nil, "NodeIPs changed for the node",
			"node", klog.KObj(node), "newNodeIPs", nodeIPs, "oldNodeIPs", n.nodeIPs)
		klog.FlushAndExit(klog.ExitFlushTimeout, 1)
	}
}

// OnNodeDelete is a handler for Node deletes.
func (n *NodeManager) OnNodeDelete(node *v1.Node) {
	n.healthServer.SyncNode(node)
	n.logger.Error(nil, "Current Node is being deleted", "node", klog.KObj(node))
}

// OnNodeSynced is a handler for Node syncs.
func (n *NodeManager) OnNodeSynced() {}
