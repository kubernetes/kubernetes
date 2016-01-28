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

	"k8s.io/kubernetes/pkg/api"
)

var emptyResource = Resource{}

// NodeInfo is node level aggregated information.
type NodeInfo struct {
	// Total requested resource of all pods (including assumed ones) on this node
	requestedResource *Resource
	pods              []*api.Pod
}

// Resource is a collection of compute resource
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
	}
	for _, pod := range pods {
		ni.addPod(pod)
	}
	return ni
}

// Pods return all pods scheduled (including assumed to be) on this node
func (n *NodeInfo) Pods() []*api.Pod {
	if n == nil {
		return nil
	}
	return n.pods
}

// RequestedResource returns aggregated resource request of pods on this node
func (n *NodeInfo) RequestedResource() Resource {
	if n == nil {
		return emptyResource
	}
	return *n.requestedResource
}

// String returns representation of human readable format of this NodeInfo.
func (n *NodeInfo) String() string {
	podKeys := make([]string, len(n.pods))
	for i, pod := range n.pods {
		podKeys[i] = pod.Name
	}
	return fmt.Sprintf("&NodeInfo{Pods:%v, RequestedResource:%#v}", podKeys, n.requestedResource)
}

// AddPod adds pod information to this NodeInfo.
func (n *NodeInfo) addPod(pod *api.Pod) {
	cpu, mem := calculateResource(pod)
	n.requestedResource.MilliCPU += cpu
	n.requestedResource.Memory += mem
	n.pods = append(n.pods, pod)
}

func calculateResource(pod *api.Pod) (int64, int64) {
	var cpu, mem int64
	for _, c := range pod.Spec.Containers {
		req := c.Resources.Requests
		cpu += req.Cpu().MilliValue()
		mem += req.Memory().Value()
	}
	return cpu, mem
}
