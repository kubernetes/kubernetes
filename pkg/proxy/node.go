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
	"reflect"
	"sync"

	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/proxy/config"
)

// NodePodCIDRHandler handles the life cycle of kube-proxy based on the node PodCIDR assigned
// Implements the config.NodeHandler interface
// https://issues.k8s.io/111321
type NodePodCIDRHandler struct {
	mu       sync.Mutex
	podCIDRs []string
}

var _ config.NodeHandler = &NodePodCIDRHandler{}

// OnNodeAdd is a handler for Node creates.
func (n *NodePodCIDRHandler) OnNodeAdd(node *v1.Node) {
	n.mu.Lock()
	defer n.mu.Unlock()

	podCIDRs := node.Spec.PodCIDRs
	// initialize podCIDRs
	if len(n.podCIDRs) == 0 && len(podCIDRs) > 0 {
		klog.InfoS("Setting current PodCIDRs", "podCIDRs", podCIDRs)
		n.podCIDRs = podCIDRs
		return
	}
	if !reflect.DeepEqual(n.podCIDRs, podCIDRs) {
		klog.ErrorS(nil, "Using NodeCIDR LocalDetector mode, current PodCIDRs are different than previous PodCIDRs, restarting",
			"node", klog.KObj(node), "newPodCIDRs", podCIDRs, "oldPodCIDRs", n.podCIDRs)
		panic("Current Node PodCIDRs are different than previous PodCIDRs, restarting")
	}
}

// OnNodeUpdate is a handler for Node updates.
func (n *NodePodCIDRHandler) OnNodeUpdate(_, node *v1.Node) {
	n.mu.Lock()
	defer n.mu.Unlock()
	podCIDRs := node.Spec.PodCIDRs
	// initialize podCIDRs
	if len(n.podCIDRs) == 0 && len(podCIDRs) > 0 {
		klog.InfoS("Setting current PodCIDRs", "podCIDRs", podCIDRs)
		n.podCIDRs = podCIDRs
		return
	}
	if !reflect.DeepEqual(n.podCIDRs, podCIDRs) {
		klog.ErrorS(nil, "Using NodeCIDR LocalDetector mode, current PodCIDRs are different than previous PodCIDRs, restarting",
			"node", klog.KObj(node), "newPodCIDRs", podCIDRs, "oldPODCIDRs", n.podCIDRs)
		panic("Current Node PodCIDRs are different than previous PodCIDRs, restarting")
	}
}

// OnNodeDelete is a handler for Node deletes.
func (n *NodePodCIDRHandler) OnNodeDelete(node *v1.Node) {
	klog.ErrorS(nil, "Current Node is being deleted", "node", klog.KObj(node))
}

// OnNodeSynced is a handler for Node syncs.
func (n *NodePodCIDRHandler) OnNodeSynced() {}
