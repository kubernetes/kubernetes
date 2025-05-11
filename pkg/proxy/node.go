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

// NodeManager handles the life cycle of kube-proxy based on the NodeIPs and PodCIDRs handles
// node watch events and crashes kube-proxy if there are any changes in NodeIPs or PodCIDRs.
// Note: It only crashes on change on PodCIDR when watchPodCIDRs is set to true.
type NodeManager struct {
	nodeInformer  v1informers.NodeInformer
	nodeLister    corelisters.NodeLister
	nodeIPs       []net.IP
	podCIDRs      []string
	watchPodCIDRs bool
	exitFunc      func(exitCode int)
}

// NewNodeManager initializes node informer that selects for the given node, waits for cache
// sync and returns NodeManager after waiting for the node object to exist and have NodeIPs
// and PodCIDRs (if watchPodCIDRs is enabled).
func NewNodeManager(ctx context.Context, client clientset.Interface,
	resyncInterval time.Duration, nodeName string, watchPodCIDRs bool,
) (*NodeManager, error) {
	// we wait for at most 5 minutes for allocators to assign a PodCIDR to the node after it is registered.
	return newNodeManager(ctx, client, resyncInterval, nodeName, watchPodCIDRs, os.Exit, time.Second, 5*time.Minute)
}

// newNodeManager implements NewNodeManager with configurable exit function, poll interval and timeout.
func newNodeManager(ctx context.Context, client clientset.Interface, resyncInterval time.Duration,
	nodeName string, watchPodCIDRs bool, exitFunc func(int), pollInterval, pollTimeout time.Duration,
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
	if !cache.WaitForNamedCacheSync("node informer cache", ctx.Done(), nodeInformer.Informer().HasSynced) {
		return nil, fmt.Errorf("can not sync node informer")
	}

	var node *v1.Node
	var err error
	var nodeIPs []net.IP
	var podCIDRs []string

	// wait for the node object to exist and have NodeIPs and PodCIDRs
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

		// we only wait for PodCIDRs if NodeManager is configured with watchPodCIDRs
		if watchPodCIDRs && len(podCIDRs) == 0 {
			if len(node.Spec.PodCIDRs) > 0 {
				podCIDRs = node.Spec.PodCIDRs
			} else {
				err = fmt.Errorf("node %q does not have any PodCIDR allocated", nodeName)
				return false, nil
			}
		}
		return true, nil
	})

	// we return the actual error in case of poll timeout
	if pollErr != nil {
		return nil, err
	}
	return &NodeManager{
		nodeInformer:  nodeInformer,
		nodeLister:    nodeLister,
		nodeIPs:       nodeIPs,
		podCIDRs:      podCIDRs,
		watchPodCIDRs: watchPodCIDRs,
		exitFunc:      exitFunc,
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
	// We exit whenever there is a change in PodCIDRs detected initially, and PodCIDRs received
	// on node watch event if the node manager is configured with watchPodCIDRs.
	if n.watchPodCIDRs {
		podCIDRs := node.Spec.PodCIDRs
		if !reflect.DeepEqual(n.podCIDRs, podCIDRs) {
			klog.InfoS("PodCIDRs changed for the node",
				"node", klog.KObj(node), "newPodCIDRs", podCIDRs, "oldPodCIDRs", n.podCIDRs)
			klog.Flush()
			n.exitFunc(1)
		}
	}

	nodeIPs, err := utilnode.GetNodeHostIPs(node)
	if err != nil {
		klog.ErrorS(err, "Failed to retrieve NodeIPs")
		return
	}

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
