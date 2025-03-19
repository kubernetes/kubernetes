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
	"k8s.io/apimachinery/pkg/util/wait"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/klog/v2"
	utilnode "k8s.io/kubernetes/pkg/util/node"
)

// NodeManager handles the life cycle of kube-proxy based on the NodeIPs, handles node watch events
// and crashes kube-proxy if there are any changes in NodeIPs.
type NodeManager struct {
	mu            sync.Mutex
	node          *v1.Node
	nodeIPs       []net.IP
	podCIDRs      []string
	watchPodCIDRs bool
	logger        klog.Logger
	exitFunc      func(exitCode int)
}

// NewNodeManager returns NodeManager after waiting for the node object to exist and have NodeIPs
// and PodCIDRS (if watchPodCIDRs is true)
func NewNodeManager(ctx context.Context, nodeLister corelisters.NodeLister, nodeName string, watchPodCIDRs bool) (*NodeManager, error) {
	// we wait for at most 5 minutes for allocators to assign a PodCIDR to the node after it is registered.
	return newNodeManager(ctx, nodeLister, nodeName, watchPodCIDRs, time.Second, 5*time.Minute)
}

// newNodeManager implements NewNodeManager with configurable poll interval and timeout.
func newNodeManager(ctx context.Context, nodeLister corelisters.NodeLister, nodeName string, watchPodCIDRs bool, pollInterval, pollTimeout time.Duration) (*NodeManager, error) {
	logger := klog.FromContext(ctx)
	var node *v1.Node
	var err error
	var nodeIPs []net.IP
	var podCIDRs []string

	ctx, cancel := context.WithTimeout(ctx, pollTimeout)
	defer cancel()
	pollErr := wait.PollUntilContextCancel(ctx, pollInterval, true, func(context.Context) (bool, error) {
		node, err = nodeLister.Get(nodeName)
		if err != nil {
			return false, nil
		}

		nodeIPs, err = utilnode.GetNodeHostIPs(node)
		if err != nil {
			err = fmt.Errorf("failed to retrieve NodeIPs : %w", err)
			return false, nil
		}

		// we only wait for PodCIDRs if NodeManager is configured with watchPodCIDRs
		if watchPodCIDRs {
			if len(node.Spec.PodCIDRs) > 0 {
				podCIDRs = node.Spec.PodCIDRs
			} else {
				err = fmt.Errorf("failed to retrieve PodCIDRs")
				return false, nil
			}
		}
		return true, nil
	})

	if pollErr != nil {
		logger.Error(err, pollErr.Error())
	} else {
		logger.Info("Successfully retrieved NodeIPs", "NodeIPs", nodeIPs)
	}

	// we return error is watchPodCIDRs was configured and PodCIDRs were not retrieved.
	if watchPodCIDRs && len(podCIDRs) == 0 {
		return nil, err
	}

	return &NodeManager{
		node:          node,
		nodeIPs:       nodeIPs,
		podCIDRs:      podCIDRs,
		watchPodCIDRs: watchPodCIDRs,
		logger:        klog.FromContext(ctx),
		exitFunc:      os.Exit,
	}, nil
}

// NodeIPs returns the NodeIPs polled in NewNodeManager().
func (n *NodeManager) NodeIPs() []net.IP {
	return n.nodeIPs
}

// PodCIDRs returns the PodCIDRs polled in NewNodeManager().
func (n *NodeManager) PodCIDRs() []string {
	return n.podCIDRs
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
		podCIDRs := node.Spec.PodCIDRs
		if !reflect.DeepEqual(n.podCIDRs, podCIDRs) {
			n.logger.Error(nil, "PodCIDRs changed for the node",
				"node", klog.KObj(node), "newPodCIDRs", podCIDRs, "oldPodCIDRs", n.podCIDRs)
			klog.Flush()
			n.exitFunc(1)
		}
	}

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

// Node returns the latest copy of node object.
func (n *NodeManager) Node() *v1.Node {
	n.mu.Lock()
	defer n.mu.Unlock()
	return n.node
}
