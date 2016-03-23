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
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/controlplane"
	"k8s.io/kubernetes/plugin/pkg/ube-scheduler/algorithm"
	priorityutil "k8s.io/kubernetes/plugin/pkg/ube-scheduler/algorithm/priorities/util"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/ube-scheduler/api"
	"k8s.io/kubernetes/plugin/pkg/ube-scheduler/schedulercache"
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
func calculateResourceOccupancy(rc *api.ReplicationController, cluster controlplane.Cluster, nodeInfo *schedulercache.ClusterInfo) schedulerapi.ClusterPriority {
	totalMilliCPU := nodeInfo.NonZeroRequest().MilliCPU
	totalMemory := nodeInfo.NonZeroRequest().Memory
	//TODO: assume Capacity is allocable resource.
	capacityMilliCPU := cluster.Status.Capacity.Cpu().MilliValue()
	capacityMemory := cluster.Status.Capacity.Memory().Value()

	// Add the resources requested by the current pod being scheduled.
	// This also helps differentiate between differently sized, but empty, nodes.
	for _, container := range rc.Spec.Template.Spec.Containers {
		cpu, memory := priorityutil.GetNonzeroRequests(&container.Resources.Requests)
		totalMilliCPU += cpu
		totalMemory += memory
	}

	cpuScore := calculateScore(totalMilliCPU, capacityMilliCPU, cluster.Name)
	memoryScore := calculateScore(totalMemory, capacityMemory, cluster.Name)
	glog.V(10).Infof(
		"%v -> %v: Least Requested Priority, Absolute/Requested: (%d, %d) / (%d, %d) Score: (%d, %d)",
		rc.Name, cluster.Name,
		totalMilliCPU, totalMemory,
		capacityMilliCPU, capacityMemory,
		cpuScore, memoryScore,
	)

	return schedulerapi.ClusterPriority{
		Cluster:  cluster.Name,
		Score: int((cpuScore + memoryScore) / 2),
	}
}

// LeastRequestedPriority is a priority function that favors nodes with fewer requested resources.
// It calculates the percentage of memory and CPU requested by pods scheduled on the node, and prioritizes
// based on the minimum of the average of the fraction of requested to capacity.
// Details: cpu((capacity - sum(requested)) * 10 / capacity) + memory((capacity - sum(requested)) * 10 / capacity) / 2
func LeastRequestedPriority(rc *api.ReplicationController, clusterNameToInfo map[string]*schedulercache.ClusterInfo, clusterLister algorithm.ClusterLister) (schedulerapi.ClusterPriorityList, error) {
	clusters, err := clusterLister.List()
	if err != nil {
		return schedulerapi.ClusterPriorityList{}, err
	}

	list := schedulerapi.ClusterPriorityList{}
	for _, node := range clusters.Items {
		list = append(list, calculateResourceOccupancy(rc, node, clusterNameToInfo[node.Name]))
	}
	return list, nil
}