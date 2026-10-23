/*
Copyright The Kubernetes Authors.

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

package preemption

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	corev1helpers "k8s.io/component-helpers/scheduling/corev1"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/helper"
)

// getHighestAllAncestor returns the key of the highest ancestor in the hierarchy that has disruption mode All.
// It returns (key, true) if found, or (empty, false) if not found.
// The search ends at the first group that cannot be resolved (a missing object, or the CompositePodGroup
// feature gate being disabled); an All ancestor found below that point still applies.
func getHighestAllAncestor(pod *v1.Pod, snapshot fwk.SharedLister) (fwk.EntityKey, bool) {
	if pod.Spec.SchedulingGroup == nil {
		return fwk.EntityKey{}, false
	}
	startKey := fwk.PodGroupKey(pod.Namespace, *pod.Spec.SchedulingGroup.PodGroupName)
	var highestAllKey fwk.EntityKey
	var hasAll bool

	for gpg, err := range helper.TraverseHierarchyUp(snapshot, startKey) {
		if err != nil {
			break
		}
		if gpg.HasDisruptionModeAll() {
			highestAllKey = gpg.GetKey()
			hasAll = true
		}
	}

	return highestAllKey, hasAll
}

// createDomainVictims converts deduplicated Victim entries into DomainVictim objects
// by enriching them with node blast-radius metadata from the snapshot, allowing the preemption
// evaluator to assess the cluster-wide impact of evicting each candidate.
func createDomainVictims(snapshot fwk.SharedLister, victims []Victim) ([]*DomainVictim, error) {
	var allPossibleVictims []*DomainVictim
	for _, vi := range victims {
		v, err := newDomainVictim(snapshot, vi.Pods(), vi.Priority(), vi.Type())
		if err != nil {
			return nil, err
		}
		allPossibleVictims = append(allPossibleVictims, v)
	}
	return allPossibleVictims, nil
}

// getWorkloadPreemptionVictims discovers all candidate victims across the cluster snapshot for workload preemption.
// Pods that are part of a pod group or composite pod group with disruption mode All are grouped
// together into a single victim. Otherwise, they are treated as individual victims.
// In both cases, the priority of the victim is the pods' own priority, which the scheduling cycle
// guarantees to match the priority of the pod group or composite pod group hierarchy they belong to.
func getWorkloadPreemptionVictims(logger klog.Logger, snapshot fwk.SharedLister) ([]Victim, error) {
	nodes, err := snapshot.NodeInfos().List()
	if err != nil {
		return nil, err
	}

	return getCrossNodesVictims(logger, snapshot, nodes), nil
}

// getCrossNodesVictims aggregates pods across the provided nodes into cluster-wide preemption candidates.
// When a pod belongs to a group hierarchy where any ancestor has disruption mode All, it groups all scheduled pods of that hierarchy
// across the cluster into a single victim so that preemption evaluates the total cluster-wide cost
// and blast radius of evicting the entire group.
func getCrossNodesVictims(logger klog.Logger, snapshot fwk.SharedLister, nodes []fwk.NodeInfo) []Victim {
	existing := sets.New[fwk.EntityKey]()
	var victims []Victim
	for _, node := range nodes {
		for _, podInfo := range node.GetPods() {
			p := podInfo.GetPod()

			highestAllKey, hasAll := getHighestAllAncestor(p, snapshot)
			if !hasAll {
				itemKey := fwk.PodKey(p.Namespace, p.Name)
				if !existing.Has(itemKey) {
					existing.Insert(itemKey)
					victims = append(victims, NewPodVictim(podInfo))
				}
				continue
			}

			if !existing.Has(highestAllKey) {
				existing.Insert(highestAllKey)
				victims = append(victims, searchCrossNodesVictimPods(logger, highestAllKey, podInfo, snapshot))
			}
		}
	}

	return victims
}

// searchCrossNodesVictimPods searches and collects all scheduled pods belonging to the leaf pod groups
// of the hierarchy starting from pgKey, and aggregates them into a single Victim.
// If no scheduled pods are found in the leaf pod groups, it falls back to creating a PodVictim for the given podInfo.
func searchCrossNodesVictimPods(
	logger klog.Logger,
	pgKey fwk.EntityKey,
	podInfo fwk.PodInfo,
	snapshot fwk.SharedLister,
) Victim {
	var allPods []*v1.Pod
	for pgState, err := range helper.GetPodGroupStates(snapshot, pgKey) {
		if err != nil {
			logger.Error(err, "Failed to get pod group state during preemption victim collection", "rootKey", pgKey)
			continue
		}
		if pgState != nil {
			allPods = append(allPods, pgState.ScheduledPods()...)
		}
	}

	if len(allPods) == 0 {
		// This should never happen since podInfo is sourced from a node in the snapshot and
		// should be present in ScheduledPods() of its leaf PodGroup. If leaf traversal or
		// PodGroupState snapshot lookups fail, treat the pod as an individual victim.
		return NewPodVictim(podInfo)
	}

	podInfos := make([]fwk.PodInfo, len(allPods))
	for i, p := range allPods {
		// pods from ScheduledPods() already passed Filter/Reserve, and cannot error here.
		podInfos[i], _ = framework.NewPodInfo(p)
	}

	// It can only return an error for empty podInfos, which is guaranteed not to be empty here.
	victim, _ := NewVictim(podInfos, corev1helpers.PodPriority(podInfo.GetPod()), pgKey.Type)
	return victim
}
