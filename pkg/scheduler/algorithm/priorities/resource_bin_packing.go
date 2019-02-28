/*
Copyright 2018 The Kubernetes Authors.

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
	"math"

	"k8s.io/api/core/v1"
	"k8s.io/klog"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	priorityutil "k8s.io/kubernetes/pkg/scheduler/algorithm/priorities/util"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
	schedulercache "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

// ResourceBinPacking contains information to calculate bin packing priority.
type ResourceBinPacking struct {
	Resources []Resources
}

//Resources contains resource name and weight
type Resources struct {
	Resource v1.ResourceName
	Weight   int
}

// NewResourceBinPacking creates a ResourceBinPackingPriorityMap.
func NewResourceBinPacking(resources []Resources) (PriorityMapFunction, PriorityReduceFunction) {
	for i, elm := range resources {
		if elm.Weight == 0 {
			resources[i].Weight = 1
		}
	}
	resourceBinPackingPrioritizer := &ResourceBinPacking{
		Resources: resources,
	}
	return resourceBinPackingPrioritizer.ResourceBinPackingPriorityMap, nil
}

// ResourceBinPackingPriorityDefault creates a resourceBinPacking based prioity
func ResourceBinPackingPriorityDefault() *ResourceBinPacking {
	defaultResourceBinPackingPrioritizer := &ResourceBinPacking{
		Resources: []Resources{},
	}
	return defaultResourceBinPackingPrioritizer
}

// ResourceBinPackingPriorityMap is a priority function that favors nodes that have higher utlization of scare resource.
// It will detect whether the requested resource is present on a node, and then calculate a score ranging from 0 to 10
// based total utlization (best fit)
// - If none of the scare resource are requested, this node will be given the lowest priority.
// - If the resource is requested, the larger the resource utlization ratio, the higher the node's priority.
func (r *ResourceBinPacking) ResourceBinPackingPriorityMap(pod *v1.Pod, meta interface{}, nodeInfo *schedulercache.NodeInfo) (schedulerapi.HostPriority, error) {
	var score int
	node := nodeInfo.Node()
	if len(r.Resources) == 0 {
		return schedulerapi.HostPriority{}, fmt.Errorf("resource not defined")
	}
	if node == nil {
		return schedulerapi.HostPriority{}, fmt.Errorf("node not found")
	}
	nodeScore := float64(0)
	weightSum := 0

	for _, elm := range r.Resources {
		if calculateResourceScore(nodeInfo, pod, elm.Resource) > 0 {
			nodeScore += calculateResourceScore(nodeInfo, pod, elm.Resource) * float64(elm.Weight)
			weightSum += elm.Weight

		}

	}
	if weightSum != 0 {
		nodeScore = nodeScore / float64(weightSum)
	}
	if klog.V(10) {
		klog.Infof(
			"%v -> %v: ResourceBinPackingPriority, Score: (%d)", pod.Name, node.Name, score,
		)
	}

	return schedulerapi.HostPriority{
		Host:  node.Name,
		Score: int(math.Round(nodeScore)),
	}, nil

}

// calculateResourceScore returns total utlization of the resource on the node
func calculateResourceScore(nodeInfo *schedulercache.NodeInfo, pod *v1.Pod, resource v1.ResourceName) float64 {
	allocatable := nodeInfo.AllocatableResource()
	used := nodeInfo.RequestedResource()
	var requested schedulercache.Resource
	var requestedInit schedulercache.Resource
	switch resource {
	case v1.ResourceCPU:
		{
			for i := range pod.Spec.Containers {
				container := &pod.Spec.Containers[i]
				cpu, _ := priorityutil.GetNonzeroRequests(&container.Resources.Requests)
				requested.MilliCPU += cpu
			}
			for i := range pod.Spec.InitContainers {
				container := &pod.Spec.InitContainers[i]
				cpu, _ := priorityutil.GetNonzeroRequests(&container.Resources.Requests)
				//As init containers are run one at a time we need to get the max requested value
				if requestedInit.MilliCPU < cpu {
					requestedInit.MilliCPU = cpu
				}
			}
			if requestedInit.MilliCPU > requested.MilliCPU {
				requested.MilliCPU = requestedInit.MilliCPU
			}

			return float64((nodeInfo.NonZeroRequest().MilliCPU+requested.MilliCPU)*schedulerapi.MaxPriority) / float64(allocatable.MilliCPU)

		}

	case v1.ResourceMemory:
		{
			for i := range pod.Spec.Containers {
				container := &pod.Spec.Containers[i]
				_, memory := priorityutil.GetNonzeroRequests(&container.Resources.Requests)
				requested.Memory += memory
			}
			for i := range pod.Spec.InitContainers {
				container := &pod.Spec.InitContainers[i]
				_, memory := priorityutil.GetNonzeroRequests(&container.Resources.Requests)
				//As init containers are run one at a time we need to get the max requested value
				if requestedInit.Memory < memory {
					requestedInit.Memory = memory
				}
			}
			if requestedInit.Memory > requested.Memory {
				requested.Memory = requestedInit.Memory
			}
			return float64((nodeInfo.NonZeroRequest().Memory+requested.Memory)*schedulerapi.MaxPriority) / float64(allocatable.Memory)
		}

	case v1.ResourceEphemeralStorage:
		{
			for i := range pod.Spec.Containers {
				container := &pod.Spec.Containers[i]
				if quantity, ok := container.Resources.Requests[v1.ResourceEphemeralStorage]; ok {
					requested.EphemeralStorage += quantity.Value()
				}

			}
			for i := range pod.Spec.InitContainers {
				container := &pod.Spec.InitContainers[i]
				if quantity, ok := container.Resources.Requests[v1.ResourceEphemeralStorage]; ok {
					//As init containers are run one at a time we need to get the max requested value
					if requestedInit.EphemeralStorage < quantity.Value() {
						requestedInit.EphemeralStorage = quantity.Value()
					}
				}
			}
			if requestedInit.EphemeralStorage > requested.EphemeralStorage {
				requested.EphemeralStorage = requestedInit.EphemeralStorage
			}
			return float64((used.EphemeralStorage+requested.EphemeralStorage)*schedulerapi.MaxPriority) / float64(allocatable.EphemeralStorage)

		}
	default:
		if v1helper.IsScalarResourceName(resource) {
			if podRequestsResource(*pod, resource) {
				for i := range pod.Spec.Containers {
					container := &pod.Spec.Containers[i]
					if quantity, ok := container.Resources.Requests[resource]; ok {
						requested.AddScalar(resource, quantity.Value())
					}
				}
				for i := range pod.Spec.InitContainers {
					container := &pod.Spec.InitContainers[i]
					if quantity, ok := container.Resources.Requests[resource]; ok {
						//As init containers are run one at a time we need to get the max requested value
						if requestedInit.ScalarResources[resource] < quantity.Value() {
							requested.SetScalar(resource, quantity.Value())
						}
					}

				}
				if requestedInit.ScalarResources[resource] > requested.ScalarResources[resource] {
					requested.ScalarResources[resource] = requestedInit.ScalarResources[resource]
				}

				return float64((used.ScalarResources[resource]+requested.ScalarResources[resource])*schedulerapi.MaxPriority) / float64(allocatable.ScalarResources[resource])

			}
		}

	}
	return 0
}

// podRequestsResource checks if a given pod requests the extended resource. if false the priority is set to 0
func podRequestsResource(pod v1.Pod, resource v1.ResourceName) bool {
	containerRequestsResource := func(container v1.Container) bool {
		for resName, quantity := range container.Resources.Requests {
			if resName == resource && quantity.MilliValue() > 0 {
				return true
			}
		}
		return false
	}

	for _, c := range pod.Spec.InitContainers {
		if containerRequestsResource(c) {
			return true
		}
	}
	for _, c := range pod.Spec.Containers {
		if containerRequestsResource(c) {
			return true
		}
	}
	return false
}
