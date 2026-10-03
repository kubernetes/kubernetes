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

package nodemanager

import (
	"context"
	"fmt"
	"net"
	"os"
	"reflect"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	v1informers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
	kubeproxyconfig "k8s.io/kubernetes/pkg/proxy/apis/config"
	proxyconfig "k8s.io/kubernetes/pkg/proxy/config"
	utilnode "k8s.io/kubernetes/pkg/util/node"
	netutils "k8s.io/utils/net"
)

// NodeManager handles the life cycle of kube-proxy based on the NodeIPs and PodCIDRs handles
// node watch events and crashes kube-proxy if there are any changes in NodeIPs or PodCIDRs.
// Note: It only crashes on change on PodCIDR when watchPodCIDRs is set to true.
type NodeManager interface {
	proxyconfig.NodeHandler

	// Name returns the node's name
	Name() string

	// PrimaryIPFamily returns the node's primary IP Family.
	PrimaryIPFamily() v1.IPFamily

	// NodeIPs returns the node's IPs. (If the NodeManager failed to get any real IPs,
	// this will return loopback IPs.)
	NodeIPs() map[v1.IPFamily]net.IP

	// PodCIDRs returns the node's PodCIDRs.
	PodCIDRs() []string

	// Node returns a copy of the latest node object, or nil if the Node has not yet
	// been seen.
	Node() *v1.Node

	// NodeInformer returns the NodeInformer.
	NodeInformer() v1informers.NodeInformer
}

type nodeManager struct {
	nodeInformer  v1informers.NodeInformer
	nodeLister    corelisters.NodeLister
	exitFunc      func(exitCode int)
	watchPodCIDRs bool

	// These are constant after construct time
	nodeName        string
	rawNodeIPs      []net.IP
	primaryIPFamily v1.IPFamily
	nodeIPs         map[v1.IPFamily]net.IP
	podCIDRs        []string

	mu   sync.Mutex
	node *v1.Node
}

var _ NodeManager = &nodeManager{}

// New initializes node informer that selects for the given node, waits for cache sync
// and returns NodeManager after waiting some amount of time for the node object to exist
// and have NodeIPs (and PodCIDRs if watchPodCIDRs is true). Note: for backward compatibility,
// it doesn't return any error if it failed to retrieve NodeIPs and watchPodCIDRs
// is false.
func New(ctx context.Context, client clientset.Interface,
	nodeName string, config *kubeproxyconfig.KubeProxyConfiguration,
) (NodeManager, error) {
	resyncInterval := config.ConfigSyncPeriod.Duration
	watchPodCIDRs := config.DetectLocalMode == kubeproxyconfig.LocalModeNodeCIDR
	nodeIPOverride := config.BindAddress
	return newNodeManager(ctx, client, resyncInterval, nodeName, nodeIPOverride, watchPodCIDRs, os.Exit, time.Second, 30*time.Second, 5*time.Minute)
}

// newNodeManager implements New with configurable exit function, poll interval and timeouts.
func newNodeManager(ctx context.Context, client clientset.Interface, resyncInterval time.Duration,
	nodeName, nodeIPOverride string, watchPodCIDRs bool, exitFunc func(int),
	pollInterval, nodeIPsTimeout, podCIDRsTimeout time.Duration,
) (*nodeManager, error) {
	// make an informer that selects for the given node
	thisNodeInformerFactory := informers.NewSharedInformerFactoryWithOptions(client, resyncInterval,
		informers.WithTransform(func(obj interface{}) (interface{}, error) {
			// kube-proxy never needs the managed fields metadata, so strip it
			// from all cached objects to reduce memory usage.
			if accessor, err := meta.Accessor(obj); err == nil {
				accessor.SetManagedFields(nil)
			}
			return obj, nil
		}),
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

	node, rawNodeIPs, podCIDRs := getNodeInfo(nodeLister, nodeName)

	if len(rawNodeIPs) == 0 {
		// wait for the node object to exist and have NodeIPs.
		ctx, cancel := context.WithTimeout(ctx, nodeIPsTimeout)
		defer cancel()
		_ = wait.PollUntilContextCancel(ctx, pollInterval, false, func(context.Context) (bool, error) {
			node, rawNodeIPs, podCIDRs = getNodeInfo(nodeLister, nodeName)
			return len(rawNodeIPs) != 0, nil
		})
	}

	if watchPodCIDRs && len(podCIDRs) == 0 {
		// wait some additional time for the PodCIDRs.
		ctx, cancel := context.WithTimeout(ctx, podCIDRsTimeout)
		defer cancel()
		_ = wait.PollUntilContextCancel(ctx, pollInterval, false, func(context.Context) (bool, error) {
			node, rawNodeIPs, podCIDRs = getNodeInfo(nodeLister, nodeName)
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
		klog.FromContext(ctx).Error(nil, "Timed out waiting for node to exist", "node", klog.KRef("", nodeName))
	} else if len(rawNodeIPs) == 0 {
		klog.FromContext(ctx).Error(nil, "Timed out waiting for node to be assigned IPs", "node", klog.KRef("", nodeName))
	}

	primaryIPFamily, nodeIPs := detectNodeIPs(rawNodeIPs, nodeIPOverride)

	return &nodeManager{
		nodeInformer:  nodeInformer,
		nodeLister:    nodeLister,
		exitFunc:      exitFunc,
		watchPodCIDRs: watchPodCIDRs,

		node:            node,
		nodeName:        nodeName,
		rawNodeIPs:      rawNodeIPs,
		primaryIPFamily: primaryIPFamily,
		nodeIPs:         nodeIPs,
		podCIDRs:        podCIDRs,
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

func detectNodeIPs(rawNodeIPs []net.IP, nodeIPOverride string) (v1.IPFamily, map[v1.IPFamily]net.IP) {
	primaryFamily := v1.IPv4Protocol
	nodeIPs := map[v1.IPFamily]net.IP{
		// default values if rawNodeIPs has no IP for either family
		v1.IPv4Protocol: net.IPv4(127, 0, 0, 1),
		v1.IPv6Protocol: net.IPv6loopback,
	}

	if len(rawNodeIPs) > 0 {
		if netutils.IsIPv6(rawNodeIPs[0]) {
			primaryFamily = v1.IPv6Protocol
		}
		nodeIPs[primaryFamily] = rawNodeIPs[0]
		if len(rawNodeIPs) > 1 {
			// If more than one address is returned, they are guaranteed to be
			// of different families
			secondaryFamily := v1.IPv4Protocol
			if netutils.IsIPv6(rawNodeIPs[1]) {
				secondaryFamily = v1.IPv6Protocol
			}
			nodeIPs[secondaryFamily] = rawNodeIPs[1]
		}
	}

	// If nodeIPOverride is passed, it overrides the primary IP
	bindIP := netutils.ParseIPSloppy(nodeIPOverride)
	if bindIP != nil && !bindIP.IsUnspecified() {
		if netutils.IsIPv4(bindIP) {
			primaryFamily = v1.IPv4Protocol
		} else {
			primaryFamily = v1.IPv6Protocol
		}
		nodeIPs[primaryFamily] = bindIP
	}

	return primaryFamily, nodeIPs
}

// Name returns the node's name
func (n *nodeManager) Name() string {
	return n.nodeName
}

// PrimaryIPFamily returns the node's primary IP Family.
func (n *nodeManager) PrimaryIPFamily() v1.IPFamily {
	return n.primaryIPFamily
}

// NodeIPs returns the node's IPs. (If the NodeManager failed to get any real IPs, this
// will return loopback IPs.)
func (n *nodeManager) NodeIPs() map[v1.IPFamily]net.IP {
	return n.nodeIPs
}

// PodCIDRs returns the node's PodCIDRs.
func (n *nodeManager) PodCIDRs() []string {
	return n.podCIDRs
}

// Node returns a copy of the latest node object, or nil if the Node has not yet been seen.
func (n *nodeManager) Node() *v1.Node {
	n.mu.Lock()
	defer n.mu.Unlock()

	if n.node == nil {
		return nil
	}
	return n.node.DeepCopy()
}

// NodeInformer returns the NodeInformer.
func (n *nodeManager) NodeInformer() v1informers.NodeInformer {
	return n.nodeInformer
}

// OnNodeChange is a handler for Node creation and update.
func (n *nodeManager) OnNodeChange(node *v1.Node) {
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
	if !reflect.DeepEqual(n.rawNodeIPs, nodeIPs) {
		klog.InfoS("NodeIPs changed for the node",
			"node", klog.KObj(node), "newNodeIPs", nodeIPs, "oldNodeIPs", n.rawNodeIPs)
		klog.Flush()
		n.exitFunc(1)
	}
}

// OnNodeDelete is a handler for Node deletes.
func (n *nodeManager) OnNodeDelete(node *v1.Node) {
	klog.InfoS("Node is being deleted", "node", klog.KObj(node))
	klog.Flush()
	n.exitFunc(1)
}

// OnNodeSynced is called after the cache is synced and all pre-existing Nodes have been reported
func (n *nodeManager) OnNodeSynced() {}
