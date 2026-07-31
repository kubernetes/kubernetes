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

package gangscheduling

import (
	"sync"

	v1 "k8s.io/api/core/v1"
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	fwk "k8s.io/kube-scheduler/framework"
)

// HierarchyTracker incrementally tracks readiness quorum counts across CompositePodGroup hierarchies.
// By maintaining quorum counts in the GangScheduling plugin, we prevent gang-scheduling policy rules
// from leaking into the core scheduler cache while supporting O(1) readiness lookups during PreEnqueue.
type HierarchyTracker interface {
	ReadyChildrenCount(key fwk.EntityKey) int

	OnPodAdd(pod *v1.Pod)
	OnPodUpdate(oldPod, newPod *v1.Pod)
	OnPodDelete(pod *v1.Pod)

	OnPodGroupAdd(pg *schedulingv1beta1.PodGroup)
	OnPodGroupUpdate(oldPG, newPG *schedulingv1beta1.PodGroup)
	OnPodGroupDelete(pg *schedulingv1beta1.PodGroup)

	OnCompositePodGroupAdd(cpg *schedulingv1alpha3.CompositePodGroup)
	OnCompositePodGroupUpdate(oldCPG, newCPG *schedulingv1alpha3.CompositePodGroup)
	OnCompositePodGroupDelete(cpg *schedulingv1alpha3.CompositePodGroup)
}

// hierarchyGroup tracks the quorum threshold and readiness state of a single PodGroup or CompositePodGroup.
// We store activePods for leaf PodGroups and readyChildren for CompositePodGroups so that readiness
// can be evaluated uniformly across different levels of the hierarchy.
type hierarchyGroup struct {
	key           fwk.EntityKey
	parentKey     *fwk.EntityKey
	minCount      int
	activePods    sets.Set[types.UID]
	readyChildren int
}

// hierarchyTrackerImpl implements HierarchyTracker with an in-memory index protected by an RWMutex
// to allow concurrent PreEnqueue lookups from scheduling worker routines.
type hierarchyTrackerImpl struct {
	mu     sync.RWMutex
	groups map[fwk.EntityKey]*hierarchyGroup
}

// NewHierarchyTracker returns a new thread-safe HierarchyTracker.
func NewHierarchyTracker() HierarchyTracker {
	return &hierarchyTrackerImpl{
		groups: make(map[fwk.EntityKey]*hierarchyGroup),
	}
}

// ReadyChildrenCount returns the number of ready child groups for a given entity key.
// We return 0 when an entity is untracked so that uninitialized or non-gang groups fail open/closed deterministically.
func (ht *hierarchyTrackerImpl) ReadyChildrenCount(key fwk.EntityKey) int {
	ht.mu.RLock()
	defer ht.mu.RUnlock()
	if group, exists := ht.groups[key]; exists {
		return group.readyChildren
	}
	return 0
}

// getOrCreateGroup retrieves or initializes a hierarchyGroup.
// We default minCount to 1 so that out-of-order informer events (e.g. a Pod arriving before its PodGroup)
// still enforce a baseline quorum until the authoritative policy definition arrives.
func (ht *hierarchyTrackerImpl) getOrCreateGroup(key fwk.EntityKey) *hierarchyGroup {
	group, exists := ht.groups[key]
	if !exists {
		group = &hierarchyGroup{
			key:        key,
			minCount:   1,
			activePods: sets.New[types.UID](),
		}
		ht.groups[key] = group
	}
	return group
}

// isGroupReady reports whether the group currently satisfies its quorum requirement.
// Leaf PodGroups evaluate quorum against active pods; CompositePodGroups evaluate against ready child groups.
func (ht *hierarchyTrackerImpl) isGroupReady(group *hierarchyGroup) bool {
	if group == nil {
		return false
	}
	if group.key.Type == fwk.PodGroupKeyType {
		return group.activePods.Len() >= group.minCount
	}
	return group.readyChildren >= group.minCount
}

// propagateReadinessDelta recursively applies a readiness delta up the group hierarchy.
// Upward propagation is triggered only when a parent transitions across its quorum boundary,
// preventing redundant updates when a group gains or loses members above or below its minCount threshold.
func (ht *hierarchyTrackerImpl) propagateReadinessDelta(parentKey *fwk.EntityKey, delta int) {
	if parentKey == nil || delta == 0 {
		return
	}
	parent := ht.getOrCreateGroup(*parentKey)
	wasReady := ht.isGroupReady(parent)
	parent.readyChildren += delta
	isNowReady := ht.isGroupReady(parent)
	if wasReady == isNowReady {
		return
	}
	propDelta := -1
	if isNowReady {
		propDelta = 1
	}
	ht.propagateReadinessDelta(parent.parentKey, propDelta)
}

// podID returns a unique identifier for a pod.
// We fall back to namespace/name when UID is empty to support test fixtures that construct pods without UIDs.
func podID(pod *v1.Pod) types.UID {
	if pod.UID != "" {
		return pod.UID
	}
	return types.UID(pod.Namespace + "/" + pod.Name)
}

// OnPodAdd records a pod as active in its scheduling group.
// When adding a pod causes its PodGroup to cross its minCount threshold, we propagate +1 readiness to the parent group.
func (ht *hierarchyTrackerImpl) OnPodAdd(pod *v1.Pod) {
	if pod == nil || pod.Spec.SchedulingGroup == nil || pod.Spec.SchedulingGroup.PodGroupName == nil {
		return
	}
	key := fwk.PodGroupKey(pod.Namespace, *pod.Spec.SchedulingGroup.PodGroupName)
	ht.mu.Lock()
	defer ht.mu.Unlock()

	group := ht.getOrCreateGroup(key)
	id := podID(pod)
	if group.activePods.Has(id) {
		return
	}
	wasReady := ht.isGroupReady(group)
	group.activePods.Insert(id)
	isNowReady := ht.isGroupReady(group)
	if wasReady == isNowReady {
		return
	}
	delta := -1
	if isNowReady {
		delta = 1
	}
	ht.propagateReadinessDelta(group.parentKey, delta)
}

// OnPodUpdate is a no-op because scheduling group membership is immutable in Kubernetes workload APIs.
func (ht *hierarchyTrackerImpl) OnPodUpdate(oldPod, newPod *v1.Pod) {
}

// OnPodDelete removes a pod from its scheduling group's active set.
// When removing a pod causes its PodGroup to drop below its minCount threshold, we propagate -1 readiness to the parent group.
func (ht *hierarchyTrackerImpl) OnPodDelete(pod *v1.Pod) {
	if pod == nil || pod.Spec.SchedulingGroup == nil || pod.Spec.SchedulingGroup.PodGroupName == nil {
		return
	}
	key := fwk.PodGroupKey(pod.Namespace, *pod.Spec.SchedulingGroup.PodGroupName)
	ht.mu.Lock()
	defer ht.mu.Unlock()

	group, exists := ht.groups[key]
	if !exists {
		return
	}
	id := podID(pod)
	if !group.activePods.Has(id) {
		return
	}
	wasReady := ht.isGroupReady(group)
	group.activePods.Delete(id)
	isNowReady := ht.isGroupReady(group)
	if wasReady == isNowReady {
		return
	}
	delta := -1
	if isNowReady {
		delta = 1
	}
	ht.propagateReadinessDelta(group.parentKey, delta)
}

// updateGroupQuorum reconciles a group's quorum threshold (minCount) and parent linkage.
// Because PodGroup or CompositePodGroup definitions may be created or updated after child pods/groups are active,
// this method reconciles readiness transitions and re-parents groups, propagating readiness deltas upwards.
func (ht *hierarchyTrackerImpl) updateGroupQuorum(key fwk.EntityKey, parentKey *fwk.EntityKey, minCount int) {
	group := ht.getOrCreateGroup(key)
	oldParent := group.parentKey
	wasReady := ht.isGroupReady(group)

	group.minCount = minCount
	group.parentKey = parentKey

	isNowReady := ht.isGroupReady(group)

	// If the parent group changed, retract readiness from the old parent and propagate to the new parent.
	if !sameParent(oldParent, parentKey) {
		if wasReady {
			ht.propagateReadinessDelta(oldParent, -1)
		}
		if isNowReady {
			ht.propagateReadinessDelta(parentKey, 1)
		}
		return
	}

	if wasReady == isNowReady {
		return
	}

	delta := -1
	if isNowReady {
		delta = 1
	}
	ht.propagateReadinessDelta(parentKey, delta)
}

// sameParent checks if two entity keys refer to the same parent group.
// We use pointer equality and value comparison to correctly handle root groups with nil parent keys.
func sameParent(a, b *fwk.EntityKey) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	return *a == *b
}

// OnPodGroupAdd registers or updates a PodGroup in the hierarchy tracker.
// We extract the gang minCount (defaulting to 1 for non-gang policies) and parent CompositePodGroup link.
func (ht *hierarchyTrackerImpl) OnPodGroupAdd(pg *schedulingv1beta1.PodGroup) {
	if pg == nil {
		return
	}
	key := fwk.PodGroupKey(pg.Namespace, pg.Name)
	minCount := 1
	if pg.Spec.SchedulingPolicy.Gang != nil {
		minCount = int(pg.Spec.SchedulingPolicy.Gang.MinCount)
	}
	var parentKey *fwk.EntityKey
	if pg.Spec.ParentCompositePodGroupName != nil {
		k := fwk.CompositePodGroupKey(pg.Namespace, *pg.Spec.ParentCompositePodGroupName)
		parentKey = &k
	}
	ht.mu.Lock()
	defer ht.mu.Unlock()
	ht.updateGroupQuorum(key, parentKey, minCount)
}

// OnPodGroupUpdate re-evaluates PodGroup quorum in case the group policy or parent link changed.
func (ht *hierarchyTrackerImpl) OnPodGroupUpdate(oldPG, newPG *schedulingv1beta1.PodGroup) {
	ht.OnPodGroupAdd(newPG)
}

// OnPodGroupDelete removes a PodGroup from tracking.
// If the PodGroup was previously ready, we retract its readiness from the parent group before deleting.
func (ht *hierarchyTrackerImpl) OnPodGroupDelete(pg *schedulingv1beta1.PodGroup) {
	if pg == nil {
		return
	}
	key := fwk.PodGroupKey(pg.Namespace, pg.Name)
	ht.mu.Lock()
	defer ht.mu.Unlock()

	group, exists := ht.groups[key]
	if !exists {
		return
	}
	wasReady := ht.isGroupReady(group)
	if wasReady {
		ht.propagateReadinessDelta(group.parentKey, -1)
	}
	delete(ht.groups, key)
}

// OnCompositePodGroupAdd registers or updates a CompositePodGroup in the hierarchy tracker.
// We extract the gang minGroupCount (defaulting to 1) and parent link to enable multi-level tree tracking.
func (ht *hierarchyTrackerImpl) OnCompositePodGroupAdd(cpg *schedulingv1alpha3.CompositePodGroup) {
	if cpg == nil {
		return
	}
	key := fwk.CompositePodGroupKey(cpg.Namespace, cpg.Name)
	minGroupCount := 1
	if cpg.Spec.SchedulingPolicy.Gang != nil {
		minGroupCount = int(cpg.Spec.SchedulingPolicy.Gang.MinGroupCount)
	}
	var parentKey *fwk.EntityKey
	if cpg.Spec.ParentCompositePodGroupName != nil {
		k := fwk.CompositePodGroupKey(cpg.Namespace, *cpg.Spec.ParentCompositePodGroupName)
		parentKey = &k
	}
	ht.mu.Lock()
	defer ht.mu.Unlock()
	ht.updateGroupQuorum(key, parentKey, minGroupCount)
}

// OnCompositePodGroupUpdate re-evaluates CompositePodGroup quorum when its policy or parent link changes.
func (ht *hierarchyTrackerImpl) OnCompositePodGroupUpdate(oldCPG, newCPG *schedulingv1alpha3.CompositePodGroup) {
	ht.OnCompositePodGroupAdd(newCPG)
}

// OnCompositePodGroupDelete removes a CompositePodGroup from tracking.
// If the CompositePodGroup was previously ready, we retract its readiness from its parent before deleting.
func (ht *hierarchyTrackerImpl) OnCompositePodGroupDelete(cpg *schedulingv1alpha3.CompositePodGroup) {
	if cpg == nil {
		return
	}
	key := fwk.CompositePodGroupKey(cpg.Namespace, cpg.Name)
	ht.mu.Lock()
	defer ht.mu.Unlock()

	group, exists := ht.groups[key]
	if !exists {
		return
	}
	wasReady := ht.isGroupReady(group)
	if wasReady {
		ht.propagateReadinessDelta(group.parentKey, -1)
	}
	delete(ht.groups, key)
}
