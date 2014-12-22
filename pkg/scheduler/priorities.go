/*
Copyright 2014 Google Inc. All rights reserved.

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

package scheduler

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/golang/glog"
)

// the unused capacity is calculated on a scale of 0-10
// 0 being the lowest priority and 10 being the highest
func calculateScore(requested, capacity int64, node string) int {
	if capacity == 0 {
		return 0
	}
	if requested > capacity {
		glog.Errorf("Combined requested resources from existing pods exceeds capacity on minion: %s", node)
		return 0
	}
	return int(((capacity - requested) * 10) / capacity)
}

// Calculate the occupancy on a node.  'node' has information about the resources on the node.
// 'pods' is a list of pods currently scheduled on the node.
func calculateOccupancy(pod api.Pod, node api.Node, pods []api.Pod) HostPriority {
	totalMilliCPU := int64(0)
	totalMemory := int64(0)
	for _, existingPod := range pods {
		for _, container := range existingPod.Spec.Containers {
			totalMilliCPU += container.CPU.MilliValue()
			totalMemory += container.Memory.Value()
		}
	}
	// Add the resources requested by the current pod being scheduled.
	// This also helps differentiate between differently sized, but empty, minions.
	for _, container := range pod.Spec.Containers {
		totalMilliCPU += container.CPU.MilliValue()
		totalMemory += container.Memory.Value()
	}

	capacityMilliCPU := node.Spec.Capacity.Get(api.ResourceCPU).MilliValue()
	capacityMemory := node.Spec.Capacity.Get(api.ResourceMemory).Value()

	cpuScore := calculateScore(totalMilliCPU, capacityMilliCPU, node.Name)
	memoryScore := calculateScore(totalMemory, capacityMemory, node.Name)
	glog.V(4).Infof(
		"%v -> %v: Least Requested Priority, AbsoluteRequested: (%d, %d) / (%d, %d) Score: (%d, %d)",
		pod.Name, node.Name,
		totalMilliCPU, totalMemory,
		capacityMilliCPU, capacityMemory,
		cpuScore, memoryScore,
	)

	return HostPriority{
		host:  node.Name,
		score: int((cpuScore + memoryScore) / 2),
	}
}

// LeastRequestedPriority is a priority function that favors nodes with fewer requested resources.
// It calculates the percentage of memory and CPU requested by pods scheduled on the node, and prioritizes
// based on the minimum of the average of the fraction of requested to capacity.
// Details: (Sum(requested cpu) / Capacity + Sum(requested memory) / Capacity) * 50
func LeastRequestedPriority(pod api.Pod, podLister PodLister, minionLister MinionLister) (HostPriorityList, error) {
	nodes, err := minionLister.List()
	if err != nil {
		return HostPriorityList{}, err
	}
	podsToMachines, err := MapPodsToMachines(podLister)

	list := HostPriorityList{}
	for _, node := range nodes.Items {
		list = append(list, calculateOccupancy(pod, node, podsToMachines[node.Name]))
	}
	return list, nil
}

type NodeLabelPrioritizer struct {
	label    string
	presence bool
}

func NewNodeLabelPriority(label string, presence bool) PriorityFunction {
	labelPrioritizer := &NodeLabelPrioritizer{
		label:    label,
		presence: presence,
	}
	return labelPrioritizer.CalculateNodeLabelPriority
}

// CalculateNodeLabelPriority checks whether a particular label exists on a minion or not, regardless of its value
// Consider the cases where the minions are places in regions/zones/racks and these are identified by labels
// In some cases, it is required that only minions that are part of ANY of the defined regions/zones/racks be selected
func (n *NodeLabelPrioritizer) CalculateNodeLabelPriority(pod api.Pod, podLister PodLister, minionLister MinionLister) (HostPriorityList, error) {
	var score int
	minions, err := minionLister.List()
	if err != nil {
		return nil, err
	}

	// find the zones that the minions belong to
	labeledMinions := map[string]bool{}
	for _, minion := range minions.Items {
		exists := labels.Set(minion.Labels).Has(n.label)
		labeledMinions[minion.Name] = (exists && n.presence) || (!exists && !n.presence)
	}

	result := []HostPriority{}
	//score int - scale of 0-10
	// 0 being the lowest priority and 10 being the highest
	for minionName, success := range labeledMinions {
		if success {
			score = 10
		} else {
			score = 0
		}
		result = append(result, HostPriority{host: minionName, score: score})
	}
	return result, nil
}
