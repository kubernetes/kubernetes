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
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/federation/apis/federation"
	"k8s.io/kubernetes/federation/plugin/pkg/scheduler/algorithm"
	priorityutil "k8s.io/kubernetes/federation/plugin/pkg/scheduler/algorithm/priorities/util"
	schedulerapi "k8s.io/kubernetes/federation/plugin/pkg/scheduler/api"
	"k8s.io/kubernetes/federation/plugin/pkg/scheduler/schedulercache"

	"github.com/golang/glog"
)

// the unused capacity is calculated on a scale of 0-10
// 0 being the lowest priority and 10 being the highest
func calculateScore(requested int64, capacity int64, cluster string) int {
	if capacity == 0 {
		return 0
	}
	if requested > capacity {
		glog.V(2).Infof("Combined requested resources %d from existing pods exceeds capacity %d on cluster %s",
			requested, capacity, cluster)
		return 0
	}
	return int(((capacity - requested) * 10) / capacity)
}

// Calculate the resource occupancy on a cluster.  'cluster' has information about the resources on the cluster.
// 'pods' is a list of pods currently scheduled on the cluster.
func calculateResourceOccupancy(rs *extensions.ReplicaSet, cluster federation.Cluster, clusterInfo *schedulercache.ClusterInfo) schedulerapi.ClusterPriority {
	totalMilliCPU := clusterInfo.NonZeroRequest().MilliCPU
	totalMemory := clusterInfo.NonZeroRequest().Memory

	capacityMilliCPU := cluster.Status.Allocatable.Cpu().MilliValue()
	capacityMemory := cluster.Status.Allocatable.Memory().Value()

	// Add the resources requested by the current rs being scheduled.
	// This also helps differentiate between differently sized, but empty, clusters.
	// TODO: discussion - the requested resources can be cache, so no need to recalculate in both predicate, prioritize and binding
	for _, container := range rs.Spec.Template.Spec.Containers {
		cpu, memory := priorityutil.GetNonzeroRequests(&container.Resources.Requests)
		totalMilliCPU += cpu
		totalMemory += memory
	}
	// Total requested resource = replicas * request per replica
	totalMemory *= int64(rs.Spec.Replicas)
	totalMilliCPU *= int64(rs.Spec.Replicas)

	cpuScore := calculateScore(totalMilliCPU, capacityMilliCPU, cluster.Name)
	memoryScore := calculateScore(totalMemory, capacityMemory, cluster.Name)
	glog.V(10).Infof(
		"%v -> %v: Least Requested Priority, Absolute/Requested: (%d, %d) / (%d, %d) Score: (%d, %d)",
		rs.Name, cluster.Name,
		totalMilliCPU, totalMemory,
		capacityMilliCPU, capacityMemory,
		cpuScore, memoryScore,
	)

	return schedulerapi.ClusterPriority{
		Cluster:  cluster.Name,
		Score: int((cpuScore + memoryScore) / 2),
	}
}

// LeastRequestedPriority is a priority function that favors clusters with fewer requested resources.
// It calculates the percentage of memory and CPU requested by pods scheduled on the cluster, and prioritizes
// based on the minimum of the average of the fraction of requested to capacity.
// Details: cpu((capacity - sum(requested)) * 10 / capacity) + memory((capacity - sum(requested)) * 10 / capacity) / 2
func LeastRequestedPriority(rc *extensions.ReplicaSet, clusterNameToInfo map[string]*schedulercache.ClusterInfo, clusterLister algorithm.ClusterLister) (schedulerapi.ClusterPriorityList, error) {
	clusters, err := clusterLister.List()
	if err != nil {
		return schedulerapi.ClusterPriorityList{}, err
	}

	list := schedulerapi.ClusterPriorityList{}
	for _, cluster := range clusters.Items {
		list = append(list, calculateResourceOccupancy(rc, cluster, clusterNameToInfo[cluster.Name]))
	}
	return list, nil
}