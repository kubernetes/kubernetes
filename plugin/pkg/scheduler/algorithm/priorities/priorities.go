/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"math"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	priorityutil "k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/priorities/util"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
)

// the unused capacity is calculated on a scale of 0-10
// 0 being the lowest priority and 10 being the highest
func calculateScore(requested int64, capacity int64, node string) int {
	if capacity == 0 {
		return 0
	}
	if requested > capacity {
		glog.V(2).Infof("Combined requested resources %d from existing pods exceeds capacity %d on node %s",
			requested, capacity, node)
		return 0
	}
	return int(((capacity - requested) * 10) / capacity)
}

// Calculate the resource occupancy on a node.  'node' has information about the resources on the node.
// 'pods' is a list of pods currently scheduled on the node.
func calculateResourceOccupancy(pod *api.Pod, node api.Node, nodeInfo *schedulercache.NodeInfo) schedulerapi.HostPriority {
	totalMilliCPU := nodeInfo.NonZeroRequest().MilliCPU
	totalMemory := nodeInfo.NonZeroRequest().Memory
	capacityMilliCPU := node.Status.Allocatable.Cpu().MilliValue()
	capacityMemory := node.Status.Allocatable.Memory().Value()

	// Add the resources requested by the current pod being scheduled.
	// This also helps differentiate between differently sized, but empty, nodes.
	for _, container := range pod.Spec.Containers {
		cpu, memory := priorityutil.GetNonzeroRequests(&container.Resources.Requests)
		totalMilliCPU += cpu
		totalMemory += memory
	}

	cpuScore := calculateScore(totalMilliCPU, capacityMilliCPU, node.Name)
	memoryScore := calculateScore(totalMemory, capacityMemory, node.Name)
	glog.V(10).Infof(
		"%v -> %v: Least Requested Priority, Absolute/Requested: (%d, %d) / (%d, %d) Score: (%d, %d)",
		pod.Name, node.Name,
		totalMilliCPU, totalMemory,
		capacityMilliCPU, capacityMemory,
		cpuScore, memoryScore,
	)

	return schedulerapi.HostPriority{
		Host:  node.Name,
		Score: int((cpuScore + memoryScore) / 2),
	}
}

// LeastRequestedPriority is a priority function that favors nodes with fewer requested resources.
// It calculates the percentage of memory and CPU requested by pods scheduled on the node, and prioritizes
// based on the minimum of the average of the fraction of requested to capacity.
// Details: cpu((capacity - sum(requested)) * 10 / capacity) + memory((capacity - sum(requested)) * 10 / capacity) / 2
func LeastRequestedPriority(pod *api.Pod, nodeNameToInfo map[string]*schedulercache.NodeInfo, nodeLister algorithm.NodeLister) (schedulerapi.HostPriorityList, error) {
	nodes, err := nodeLister.List()
	if err != nil {
		return schedulerapi.HostPriorityList{}, err
	}

	list := schedulerapi.HostPriorityList{}
	for _, node := range nodes.Items {
		list = append(list, calculateResourceOccupancy(pod, node, nodeNameToInfo[node.Name]))
	}
	return list, nil
}

type NodeLabelPrioritizer struct {
	label    string
	presence bool
}

func NewNodeLabelPriority(label string, presence bool) algorithm.PriorityFunction {
	labelPrioritizer := &NodeLabelPrioritizer{
		label:    label,
		presence: presence,
	}
	return labelPrioritizer.CalculateNodeLabelPriority
}

// CalculateNodeLabelPriority checks whether a particular label exists on a node or not, regardless of its value.
// If presence is true, prioritizes nodes that have the specified label, regardless of value.
// If presence is false, prioritizes nodes that do not have the specified label.
func (n *NodeLabelPrioritizer) CalculateNodeLabelPriority(pod *api.Pod, nodeNameToInfo map[string]*schedulercache.NodeInfo, nodeLister algorithm.NodeLister) (schedulerapi.HostPriorityList, error) {
	var score int
	nodes, err := nodeLister.List()
	if err != nil {
		return nil, err
	}

	labeledNodes := map[string]bool{}
	for _, node := range nodes.Items {
		exists := labels.Set(node.Labels).Has(n.label)
		labeledNodes[node.Name] = (exists && n.presence) || (!exists && !n.presence)
	}

	result := []schedulerapi.HostPriority{}
	//score int - scale of 0-10
	// 0 being the lowest priority and 10 being the highest
	for nodeName, success := range labeledNodes {
		if success {
			score = 10
		} else {
			score = 0
		}
		result = append(result, schedulerapi.HostPriority{Host: nodeName, Score: score})
	}
	return result, nil
}

// This is a reasonable size range of all container images. 90%ile of images on dockerhub drops into this range.
const (
	mb         int64 = 1024 * 1024
	minImgSize int64 = 23 * mb
	maxImgSize int64 = 1000 * mb
)

// ImageLocalityPriority is a priority function that favors nodes that already have requested pod container's images.
// It will detect whether the requested images are present on a node, and then calculate a score ranging from 0 to 10
// based on the total size of those images.
// - If none of the images are present, this node will be given the lowest priority.
// - If some of the images are present on a node, the larger their sizes' sum, the higher the node's priority.
func ImageLocalityPriority(pod *api.Pod, nodeNameToInfo map[string]*schedulercache.NodeInfo, nodeLister algorithm.NodeLister) (schedulerapi.HostPriorityList, error) {
	sumSizeMap := make(map[string]int64)

	nodes, err := nodeLister.List()
	if err != nil {
		return nil, err
	}

	for _, container := range pod.Spec.Containers {
		for _, node := range nodes.Items {
			// Check if this container's image is present and get its size.
			imageSize := checkContainerImageOnNode(node, container)
			// Add this size to the total result of this node.
			sumSizeMap[node.Name] += imageSize
		}
	}

	result := []schedulerapi.HostPriority{}
	// score int - scale of 0-10
	// 0 being the lowest priority and 10 being the highest.
	for nodeName, sumSize := range sumSizeMap {
		result = append(result, schedulerapi.HostPriority{Host: nodeName,
			Score: calculateScoreFromSize(sumSize)})
	}
	return result, nil
}

// checkContainerImageOnNode checks if a container image is present on a node and returns its size.
func checkContainerImageOnNode(node api.Node, container api.Container) int64 {
	for _, image := range node.Status.Images {
		for _, name := range image.Names {
			if container.Image == name {
				// Should return immediately.
				return image.SizeBytes
			}
		}
	}
	return 0
}

// calculateScoreFromSize calculates the priority of a node. sumSize is sum size of requested images on this node.
// 1. Split image size range into 10 buckets.
// 2. Decide the priority of a given sumSize based on which bucket it belongs to.
func calculateScoreFromSize(sumSize int64) int {
	var score int
	switch {
	case sumSize == 0 || sumSize < minImgSize:
		// score == 0 means none of the images required by this pod are present on this
		// node or the total size of the images present is too small to be taken into further consideration.
		score = 0
	// If existing images' total size is larger than max, just make it highest priority.
	case sumSize >= maxImgSize:
		score = 10
	default:
		score = int((10 * (sumSize - minImgSize) / (maxImgSize - minImgSize)) + 1)
	}
	// Return which bucket the given size belongs to
	return score
}

// BalancedResourceAllocation favors nodes with balanced resource usage rate.
// BalancedResourceAllocation should **NOT** be used alone, and **MUST** be used together with LeastRequestedPriority.
// It calculates the difference between the cpu and memory fracion of capacity, and prioritizes the host based on how
// close the two metrics are to each other.
// Detail: score = 10 - abs(cpuFraction-memoryFraction)*10. The algorithm is partly inspired by:
// "Wei Huang et al. An Energy Efficient Virtual Machine Placement Algorithm with Balanced Resource Utilization"
func BalancedResourceAllocation(pod *api.Pod, nodeNameToInfo map[string]*schedulercache.NodeInfo, nodeLister algorithm.NodeLister) (schedulerapi.HostPriorityList, error) {
	nodes, err := nodeLister.List()
	if err != nil {
		return schedulerapi.HostPriorityList{}, err
	}

	list := schedulerapi.HostPriorityList{}
	for _, node := range nodes.Items {
		list = append(list, calculateBalancedResourceAllocation(pod, node, nodeNameToInfo[node.Name]))
	}
	return list, nil
}

func calculateBalancedResourceAllocation(pod *api.Pod, node api.Node, nodeInfo *schedulercache.NodeInfo) schedulerapi.HostPriority {
	totalMilliCPU := nodeInfo.NonZeroRequest().MilliCPU
	totalMemory := nodeInfo.NonZeroRequest().Memory
	score := int(0)
	// Add the resources requested by the current pod being scheduled.
	// This also helps differentiate between differently sized, but empty, nodes.
	for _, container := range pod.Spec.Containers {
		cpu, memory := priorityutil.GetNonzeroRequests(&container.Resources.Requests)
		totalMilliCPU += cpu
		totalMemory += memory
	}

	capacityMilliCPU := node.Status.Allocatable.Cpu().MilliValue()
	capacityMemory := node.Status.Allocatable.Memory().Value()

	cpuFraction := fractionOfCapacity(totalMilliCPU, capacityMilliCPU)
	memoryFraction := fractionOfCapacity(totalMemory, capacityMemory)
	if cpuFraction >= 1 || memoryFraction >= 1 {
		// if requested >= capacity, the corresponding host should never be preferrred.
		score = 0
	} else {
		// Upper and lower boundary of difference between cpuFraction and memoryFraction are -1 and 1
		// respectively. Multilying the absolute value of the difference by 10 scales the value to
		// 0-10 with 0 representing well balanced allocation and 10 poorly balanced. Subtracting it from
		// 10 leads to the score which also scales from 0 to 10 while 10 representing well balanced.
		diff := math.Abs(cpuFraction - memoryFraction)
		score = int(10 - diff*10)
	}
	glog.V(10).Infof(
		"%v -> %v: Balanced Resource Allocation, Absolute/Requested: (%d, %d) / (%d, %d) Score: (%d)",
		pod.Name, node.Name,
		totalMilliCPU, totalMemory,
		capacityMilliCPU, capacityMemory,
		score,
	)

	return schedulerapi.HostPriority{
		Host:  node.Name,
		Score: score,
	}
}

func fractionOfCapacity(requested, capacity int64) float64 {
	if capacity == 0 {
		return 1
	}
	return float64(requested) / float64(capacity)
}
