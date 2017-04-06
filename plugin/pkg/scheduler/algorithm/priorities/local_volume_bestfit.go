/*
Copyright 2014 The Kubernetes Authors.

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

	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/predicates"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
)

const megabytes int = 1024 * 1024

type localPVBestFit struct {
	pvInfo  predicates.PersistentVolumeInfo
	pvcInfo predicates.PersistentVolumeClaimInfo
	client  clientset.Interface
}

func NewLocalPVBestFit(pvInfo predicates.PersistentVolumeInfo, pvcInfo predicates.PersistentVolumeClaimInfo, client clientset.Interface) (algorithm.PriorityMapFunction, algorithm.PriorityReduceFunction) {
	c := &localPVBestFit{
		pvInfo:  pvInfo,
		pvcInfo: pvcInfo,
		client:  client,
	}
	return c.BestNodeForLocalPVRequestsMap, c.BestNodeForLocalPVRequestsReduce
}

func (c *localPVBestFit) BestNodeForLocalPVRequestsMap(pod *v1.Pod, meta interface{}, nodeInfo *schedulercache.NodeInfo) (schedulerapi.HostPriority, error) {
	node := nodeInfo.Node()

	result := schedulerapi.HostPriority{
		Host:  node.Name,
		Score: 0,
	}
	// If a pod doesn't have any volume attached to it, the predicate will always be true.
	// Thus we make a fast path for it, to avoid unnecessary computations in this case.
	if len(pod.Spec.Volumes) == 0 {
		return result, nil
	}

	unboundClaims, err := predicates.GetUnboundClaims(pod, c.pvcInfo)
	if err != nil {
		return schedulerapi.HostPriority{}, fmt.Errorf("Failed to get list of unbound Persistent Volume Claims: %v", err)
	}

	claimVolumeBindings, err := predicates.GetVolumesForUnboundClaims(unboundClaims, node, c.pvInfo)
	if err != nil {
		return schedulerapi.HostPriority{}, err
	}
	wasted := 0
	for _, claimVolumeBinding := range claimVolumeBindings {
		requestedQuantity := claimVolumeBinding.Claim.Spec.Resources.Requests[v1.ResourceStorage]
		allocatedQuantity := claimVolumeBinding.Volume.Spec.Capacity[v1.ResourceStorage]
		// Normalize to megabytes to fit in an integer.
		wasted += int(allocatedQuantity.Value()-requestedQuantity.Value()) / megabytes
	}
	result.Score = wasted
	return result, nil
}

func (c *localPVBestFit) BestNodeForLocalPVRequestsReduce(pod *v1.Pod, meta interface{}, nodeNameToInfo map[string]*schedulercache.NodeInfo, result schedulerapi.HostPriorityList) error {
	var minwastage int
	for i := range result {
		if result[i].Score < minwastage {
			minwastage = result[i].Score
		}
	}
	minwastageFloat := float64(minwastage)

	for i := range result {
		result[i].Score = int(10 - (float64(result[i].Score) / minwastageFloat))
	}
	return nil
}
