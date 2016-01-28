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

import "k8s.io/kubernetes/pkg/api"

// NodeInfo is node level aggregated information.
type NodeInfo struct {
	// Total requested resource of all pods (including assumed ones) on this node
	RequestedResource *Resource
	Pods              []*api.Pod
}

// Resource is a collection of compute resource
type Resource struct {
	MilliCPU int64
	Memory   int64
}

// NewNodeInfo returns a ready to use NodeInfo object.
func NewNodeInfo() *NodeInfo {
	return &NodeInfo{
		RequestedResource: &Resource{},
	}
}

// AddPod adds pod information to this NodeInfo.
func (n *NodeInfo) AddPod(pod *api.Pod) {
	cpu, mem := calculateResource(pod)
	n.RequestedResource.MilliCPU += cpu
	n.RequestedResource.Memory += mem
	n.Pods = append(n.Pods, pod)
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
