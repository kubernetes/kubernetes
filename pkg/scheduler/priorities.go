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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/resources"
	"github.com/golang/glog"
)

func calculatePercentage(requested, capacity int) int {
	if capacity == 0 {
		return 0
	}
	return (requested * 100) / capacity
}

// Calculate the occupancy on a node.  'node' has information about the resources on the node.
// 'pods' is a list of pods currently scheduled on the node.
func calculateOccupancy(node api.Minion, pods []api.Pod) HostPriority {
	totalCPU := 0
	totalMemory := 0
	for _, pod := range pods {
		for _, container := range pod.DesiredState.Manifest.Containers {
			totalCPU += container.CPU
			totalMemory += container.Memory
		}
	}

	percentageCPU := calculatePercentage(totalCPU, resources.GetIntegerResource(node.NodeResources.Capacity, resources.CPU, 0))
	percentageMemory := calculatePercentage(totalMemory, resources.GetIntegerResource(node.NodeResources.Capacity, resources.Memory, 0))
	glog.V(4).Infof("Least Requested Priority, AbsoluteRequested: (%d, %d) Percentage:(%d\\%m, %d\\%)", totalCPU, totalMemory, percentageCPU, percentageMemory)

	return HostPriority{
		host:  node.Name,
		score: int((percentageCPU + percentageMemory) / 2),
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
		list = append(list, calculateOccupancy(node, podsToMachines[node.Name]))
	}
	return list, nil
}
