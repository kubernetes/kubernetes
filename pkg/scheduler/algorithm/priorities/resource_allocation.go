/*
Copyright 2017 The Kubernetes Authors.

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

package priorities

import (
	"fmt"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	priorityutil "k8s.io/kubernetes/pkg/scheduler/algorithm/priorities/util"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
	"k8s.io/kubernetes/pkg/scheduler/schedulercache"
)

type ResourceAllocationPriority struct {
	Name   string
	scorer func(requested, allocable *schedulercache.Resource) int64
}

func (r *ResourceAllocationPriority) PriorityMap(
	pod *v1.Pod,
	meta interface{},
	nodeInfo *schedulercache.NodeInfo) (schedulerapi.HostPriority, error) {
	node := nodeInfo.Node()
	if node == nil {
		return schedulerapi.HostPriority{}, fmt.Errorf("node not found")
	}
	allocatable := nodeInfo.AllocatableResource()

	var requested schedulercache.Resource
	if priorityMeta, ok := meta.(*priorityMetadata); ok {
		requested = *priorityMeta.nonZeroRequest
	} else {
		// We couldn't parse metadatat - fallback to computing it.
		requested = *getNonZeroRequests(pod)
	}

	requested.MilliCPU += nodeInfo.NonZeroRequest().MilliCPU
	requested.Memory += nodeInfo.NonZeroRequest().Memory

	score := r.scorer(&requested, &allocatable)

	if glog.V(10) {
		glog.Infof(
			"%v -> %v: %v, capacity %d millicores %d memory bytes, total request %d millicores %d memory bytes, score %d",
			pod.Name, node.Name, r.Name,
			allocatable.MilliCPU, allocatable.Memory,
			requested.MilliCPU+allocatable.MilliCPU, requested.Memory+allocatable.Memory,
			score,
		)
	}

	return schedulerapi.HostPriority{
		Host:  node.Name,
		Score: int(score),
	}, nil
}

func getNonZeroRequests(pod *v1.Pod) *schedulercache.Resource {
	result := &schedulercache.Resource{}
	for i := range pod.Spec.Containers {
		container := &pod.Spec.Containers[i]
		cpu, memory := priorityutil.GetNonzeroRequests(&container.Resources.Requests)
		result.MilliCPU += cpu
		result.Memory += memory
	}
	return result
}
