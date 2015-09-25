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
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/predicates"
)

// the unused capacity is calculated on a scale of 0-10
// 0 being the lowest priority and 10 being the highest
func calculateScore(requested int64, capacity int64, node string) int {
	if capacity == 0 {
		return 0
	}
	if requested > capacity {
		glog.Infof("Combined requested resources %d from existing pods exceeds capacity %d on node %s",
			requested, capacity, node)
		return 0
	}
	return int(((capacity - requested) * 10) / capacity)
}

// For each of these resources, a pod that doesn't request the resource explicitly
// will be treated as having requested the amount indicated below, for the purpose
// of computing priority only. This ensures that when scheduling zero-request pods, such
// pods will not all be scheduled to the machine with the smallest in-use request,
// and that when scheduling regular pods, such pods will not see zero-request pods as
// consuming no resources whatsoever. We chose these values to be similar to the
// resources that we give to cluster addon pods (#10653). But they are pretty arbitrary.
// As described in #11713, we use request instead of limit to deal with resource requirements.
const defaultMilliCpuRequest int64 = 100             // 0.1 core
const defaultMemoryRequest int64 = 200 * 1024 * 1024 // 200 MB

// TODO: Consider setting default as a fixed fraction of machine capacity (take "capacity api.ResourceList"
// as an additional argument here) rather than using constants
func getNonzeroRequests(requests *api.ResourceList) (int64, int64) {
	var out_millicpu, out_memory int64
	// Override if un-set, but not if explicitly set to zero
	if (*requests.Cpu() == resource.Quantity{}) {
		out_millicpu = defaultMilliCpuRequest
	} else {
		out_millicpu = requests.Cpu().MilliValue()
	}
	// Override if un-set, but not if explicitly set to zero
	if (*requests.Memory() == resource.Quantity{}) {
		out_memory = defaultMemoryRequest
	} else {
		out_memory = requests.Memory().Value()
	}
	return out_millicpu, out_memory
}

// Calculate the resource occupancy on a node.  'node' has information about the resources on the node.
// 'pods' is a list of pods currently scheduled on the node.
func calculateResourceOccupancy(pod *api.Pod, node api.Node, pods []*api.Pod) algorithm.HostPriority {
	totalMilliCPU := int64(0)
	totalMemory := int64(0)
	capacityMilliCPU := node.Status.Capacity.Cpu().MilliValue()
	capacityMemory := node.Status.Capacity.Memory().Value()

	for _, existingPod := range pods {
		for _, container := range existingPod.Spec.Containers {
			cpu, memory := getNonzeroRequests(&container.Resources.Requests)
			totalMilliCPU += cpu
			totalMemory += memory
		}
	}
	// Add the resources requested by the current pod being scheduled.
	// This also helps differentiate between differently sized, but empty, nodes.
	for _, container := range pod.Spec.Containers {
		cpu, memory := getNonzeroRequests(&container.Resources.Requests)
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

	return algorithm.HostPriority{
		Host:  node.Name,
		Score: int((cpuScore + memoryScore) / 2),
	}
}

// LeastRequestedPriority is a priority function that favors nodes with fewer requested resources.
// It calculates the percentage of memory and CPU requested by pods scheduled on the node, and prioritizes
// based on the minimum of the average of the fraction of requested to capacity.
// Details: cpu((capacity - sum(requested)) * 10 / capacity) + memory((capacity - sum(requested)) * 10 / capacity) / 2
func LeastRequestedPriority(pod *api.Pod, podLister algorithm.PodLister, nodeLister algorithm.NodeLister) (algorithm.HostPriorityList, error) {
	nodes, err := nodeLister.List()
	if err != nil {
		return algorithm.HostPriorityList{}, err
	}
	podsToMachines, err := predicates.MapPodsToMachines(podLister)

	list := algorithm.HostPriorityList{}
	for _, node := range nodes.Items {
		list = append(list, calculateResourceOccupancy(pod, node, podsToMachines[node.Name]))
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
func (n *NodeLabelPrioritizer) CalculateNodeLabelPriority(pod *api.Pod, podLister algorithm.PodLister, nodeLister algorithm.NodeLister) (algorithm.HostPriorityList, error) {
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

	result := []algorithm.HostPriority{}
	//score int - scale of 0-10
	// 0 being the lowest priority and 10 being the highest
	for nodeName, success := range labeledNodes {
		if success {
			score = 10
		} else {
			score = 0
		}
		result = append(result, algorithm.HostPriority{Host: nodeName, Score: score})
	}
	return result, nil
}

// BalancedResourceAllocation favors nodes with balanced resource usage rate.
// BalancedResourceAllocation should **NOT** be used alone, and **MUST** be used together with LeastRequestedPriority.
// It calculates the difference between the cpu and memory fracion of capacity, and prioritizes the host based on how
// close the two metrics are to each other.
// Detail: score = 10 - abs(cpuFraction-memoryFraction)*10. The algorithm is partly inspired by:
// "Wei Huang et al. An Energy Efficient Virtual Machine Placement Algorithm with Balanced Resource Utilization"
func BalancedResourceAllocation(pod *api.Pod, podLister algorithm.PodLister, nodeLister algorithm.NodeLister) (algorithm.HostPriorityList, error) {
	nodes, err := nodeLister.List()
	if err != nil {
		return algorithm.HostPriorityList{}, err
	}
	podsToMachines, err := predicates.MapPodsToMachines(podLister)

	list := algorithm.HostPriorityList{}
	for _, node := range nodes.Items {
		list = append(list, calculateBalancedResourceAllocation(pod, node, podsToMachines[node.Name]))
	}
	return list, nil
}

func calculateBalancedResourceAllocation(pod *api.Pod, node api.Node, pods []*api.Pod) algorithm.HostPriority {
	totalMilliCPU := int64(0)
	totalMemory := int64(0)
	score := int(0)
	for _, existingPod := range pods {
		for _, container := range existingPod.Spec.Containers {
			cpu, memory := getNonzeroRequests(&container.Resources.Requests)
			totalMilliCPU += cpu
			totalMemory += memory
		}
	}
	// Add the resources requested by the current pod being scheduled.
	// This also helps differentiate between differently sized, but empty, nodes.
	for _, container := range pod.Spec.Containers {
		cpu, memory := getNonzeroRequests(&container.Resources.Requests)
		totalMilliCPU += cpu
		totalMemory += memory
	}

	capacityMilliCPU := node.Status.Capacity.Cpu().MilliValue()
	capacityMemory := node.Status.Capacity.Memory().Value()

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

	return algorithm.HostPriority{
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
