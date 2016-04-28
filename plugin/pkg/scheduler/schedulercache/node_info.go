/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package schedulercache

import (
	"fmt"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api"
	clientcache "k8s.io/kubernetes/pkg/client/cache"
	priorityutil "k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/priorities/util"
)

var emptyResource = Resource{}

// NodeInfo is node level aggregated information.
type NodeInfo struct {
	// Overall node information.
	node *api.Node

	// Total requested resource of all pods on this node.
	// It includes assumed pods which scheduler sends binding to apiserver but
	// didn't get it as scheduled yet.
	requestedResource *Resource
	pods              []*api.Pod
	nonzeroRequest    *Resource
}

// Resource is a collection of compute resource.
type Resource struct {
	MilliCPU int64
	Memory   int64
}

// NewNodeInfo returns a ready to use empty NodeInfo object.
// If any pods are given in arguments, their information will be aggregated in
// the returned object.
func NewNodeInfo(pods ...*api.Pod) *NodeInfo {
	ni := &NodeInfo{
		requestedResource: &Resource{},
		nonzeroRequest:    &Resource{},
	}
	for _, pod := range pods {
		ni.addPod(pod)
	}
	return ni
}

// Returns overall information about this node.
func (n *NodeInfo) Node() *api.Node {
	if n == nil {
		return nil
	}
	return n.node
}

// Pods return all pods scheduled (including assumed to be) on this node.
func (n *NodeInfo) Pods() []*api.Pod {
	if n == nil {
		return nil
	}
	return n.pods
}

// RequestedResource returns aggregated resource request of pods on this node.
func (n *NodeInfo) RequestedResource() Resource {
	if n == nil {
		return emptyResource
	}
	return *n.requestedResource
}

// NonZeroRequest returns aggregated nonzero resource request of pods on this node.
func (n *NodeInfo) NonZeroRequest() Resource {
	if n == nil {
		return emptyResource
	}
	return *n.nonzeroRequest
}

func (n *NodeInfo) Clone() *NodeInfo {
	pods := append([]*api.Pod(nil), n.pods...)
	clone := &NodeInfo{
		node:              n.node,
		requestedResource: &(*n.requestedResource),
		nonzeroRequest:    &(*n.nonzeroRequest),
		pods:              pods,
	}
	return clone
}

// String returns representation of human readable format of this NodeInfo.
func (n *NodeInfo) String() string {
	podKeys := make([]string, len(n.pods))
	for i, pod := range n.pods {
		podKeys[i] = pod.Name
	}
	return fmt.Sprintf("&NodeInfo{Pods:%v, RequestedResource:%#v, NonZeroRequest: %#v}", podKeys, n.requestedResource, n.nonzeroRequest)
}

// addPod adds pod information to this NodeInfo.
func (n *NodeInfo) addPod(pod *api.Pod) {
	cpu, mem, non0_cpu, non0_mem := calculateResource(pod)
	n.requestedResource.MilliCPU += cpu
	n.requestedResource.Memory += mem
	n.nonzeroRequest.MilliCPU += non0_cpu
	n.nonzeroRequest.Memory += non0_mem
	n.pods = append(n.pods, pod)
}

// removePod subtracts pod information to this NodeInfo.
func (n *NodeInfo) removePod(pod *api.Pod) error {
	k1, err := getPodKey(pod)
	if err != nil {
		return err
	}

	cpu, mem, non0_cpu, non0_mem := calculateResource(pod)
	n.requestedResource.MilliCPU -= cpu
	n.requestedResource.Memory -= mem
	n.nonzeroRequest.MilliCPU -= non0_cpu
	n.nonzeroRequest.Memory -= non0_mem

	for i := range n.pods {
		k2, err := getPodKey(n.pods[i])
		if err != nil {
			glog.Errorf("Cannot get pod key, err: %v", err)
			continue
		}
		if k1 == k2 {
			// delete the element
			n.pods[i] = n.pods[len(n.pods)-1]
			n.pods = n.pods[:len(n.pods)-1]
			return nil
		}
	}
	return fmt.Errorf("no corresponding pod in pods")
}

func calculateResource(pod *api.Pod) (cpu int64, mem int64, non0_cpu int64, non0_mem int64) {
	for _, c := range pod.Spec.Containers {
		req := c.Resources.Requests
		cpu += req.Cpu().MilliValue()
		mem += req.Memory().Value()

		non0_cpu_req, non0_mem_req := priorityutil.GetNonzeroRequests(&req)
		non0_cpu += non0_cpu_req
		non0_mem += non0_mem_req
	}
	return
}

// Sets the overall node information.
func (n *NodeInfo) SetNode(node *api.Node) error {
	n.node = node
	return nil
}

// Removes the overall information about the node.
func (n *NodeInfo) RemoveNode(node *api.Node) error {
	// We don't remove NodeInfo for because there can still be some pods on this node -
	// this is because notifications about pods are delivered in a different watch,
	// and thus can potentially be observed later, even though they happened before
	// node removal. This is handled correctly in cache.go file.
	n.node = nil
	return nil
}

// getPodKey returns the string key of a pod.
func getPodKey(pod *api.Pod) (string, error) {
	return clientcache.MetaNamespaceKeyFunc(pod)
}
