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

// NodeInfo is node level aggregated information.
// Note: if change fields, also change String() method
type NodeInfo struct {
	PodNum int
	// Total requested resource of all pods (including assumed ones) on this node
	RequestedResource *Resource
}

// Resource is a collection of compute resource .
type Resource struct {
	MilliCPU int64
	Memory   int64
}

// PodInfo is *internal* struct used to collect and transfer pod's information to NodeInfo.
// It's also used by assumedPod struct to keep local copy of information.
type PodInfo struct {
	RequestedResource Resource
}

// ParsePodInfo will parse an api.Pod struct to PodInfo.
func ParsePodInfo(pod *api.Pod) PodInfo {
	var cpu, mem int64
	for _, c := range pod.Spec.Containers {
		req := c.Resources.Requests
		cpu += req.Cpu().MilliValue()
		mem += req.Memory().Value()
	}
	return PodInfo{
		RequestedResource: Resource{
			MilliCPU: cpu,
			Memory:   mem,
		},
	}
}

// NewNodeInfo returns a ready to use NodeInfo object.
func NewNodeInfo() *NodeInfo {
	return &NodeInfo{
		RequestedResource: &Resource{},
		PodNum:            0,
	}
}

// AddPodInfo adds pod information to this NodeInfo.
func (n *NodeInfo) AddPodInfo(pi PodInfo) {
	n.RequestedResource.MilliCPU += pi.RequestedResource.MilliCPU
	n.RequestedResource.Memory += pi.RequestedResource.Memory
	n.PodNum++
}

// RemovePodInfo subtracts pod information to this NodeInfo.
func (n *NodeInfo) RemovePodInfo(pi PodInfo) {
	n.RequestedResource.MilliCPU -= pi.RequestedResource.MilliCPU
	n.RequestedResource.Memory -= pi.RequestedResource.Memory
	n.PodNum--
}

// String returns representation of human readable format of this NodeInfo.
func (n *NodeInfo) String() string {
	return fmt.Sprintf("&NodeInfo{PodNum:%v, RequestedResource:%#v}", n.PodNum, n.RequestedResource)
}
