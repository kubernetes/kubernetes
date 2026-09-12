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
	"k8s.io/utils/ptr"
)

// hierarchyGroup tracks the readiness quorum (minCount / minGroupCount) and state of a single PodGroup or CompositePodGroup.
// We store activePods for leaf PodGroups and readyChildren for CompositePodGroups so that readiness
// can be evaluated uniformly across different levels of the hierarchy.
type hierarchyGroup struct {
	key           fwk.EntityKey
	parentKey     *fwk.EntityKey
	minCount      int
	activePods    sets.Set[types.UID]
	readyChildren int
}

// hierarchyTrackerImpl implements fwk.PodGroupHierarchyTracker with an in-memory index protected by an RWMutex
// to allow concurrent PreEnqueue lookups from scheduling worker routines.
type hierarchyTrackerImpl struct {
	lock   sync.RWMutex
	groups map[fwk.EntityKey]*hierarchyGroup
}

var _ fwk.PodGroupHierarchyTracker = &hierarchyTrackerImpl{}

// NewHierarchyTracker returns a new thread-safe PodGroupHierarchyTracker.
func NewHierarchyTracker() fwk.PodGroupHierarchyTracker {
	return &hierarchyTrackerImpl{
		groups: make(map[fwk.EntityKey]*hierarchyGroup),
	}
}

// ReadyChildrenCount returns the number of ready child groups for a given entity key.
// We return 0 when an entity is untracked so that uninitialized or non-gang groups fail open/closed deterministically.
func (ht *hierarchyTrackerImpl) ReadyChildrenCount(key fwk.EntityKey) int {
	ht.lock.RLock()
	defer ht.lock.RUnlock()
	if group, exists := ht.groups[key]; exists {
		return group.readyChildren
	}
	return 0
}

// getOrCreateGroup retrieves or initializes a hierarchyGroup. It assumes ht.lock is held.
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

// isGroupReady reports whether the group currently satisfies its readiness requirement. It assumes ht.lock is held.
// Leaf PodGroups evaluate readiness against active pods; CompositePodGroups evaluate against ready child groups.
func (ht *hierarchyTrackerImpl) isGroupReady(group *hierarchyGroup) bool {
	if group == nil {
		return false
	}
	if group.key.Type == fwk.PodGroupKeyType {
		return group.activePods.Len() >= group.minCount
	}
	return group.readyChildren >= group.minCount
}

// propagateReadinessDelta recursively applies a readiness delta up the group hierarchy. It assumes ht.lock is held.
// Upward propagation is triggered only when a parent transitions across its readiness boundary,
// preventing redundant updates when a group gains or loses members above or below its minCount quorum.
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

// podUID returns a unique identifier for a pod.
// We fall back to namespace/name when UID is empty to support test fixtures that construct pods without UIDs.
func podUID(pod *v1.Pod) types.UID {
	if pod.UID != "" {
		return pod.UID
	}
	return types.UID(pod.Namespace + "/" + pod.Name)
}

// OnPodAdd records a pod as active in its scheduling group.
// When adding a pod causes its PodGroup to cross its minCount quorum, we propagate +1 readiness to the parent group.
func (ht *hierarchyTrackerImpl) OnPodAdd(pod *v1.Pod) {
	if pod == nil || pod.Spec.SchedulingGroup == nil || pod.Spec.SchedulingGroup.PodGroupName == nil {
		return
	}
	key := fwk.PodGroupKey(pod.Namespace, *pod.Spec.SchedulingGroup.PodGroupName)
	ht.lock.Lock()
	defer ht.lock.Unlock()

	group := ht.getOrCreateGroup(key)
	uid := podUID(pod)
	if group.activePods.Has(uid) {
		return
	}
	wasReady := ht.isGroupReady(group)
	group.activePods.Insert(uid)
	if !wasReady && ht.isGroupReady(group) {
		ht.propagateReadinessDelta(group.parentKey, 1)
	}
}

// OnPodUpdate is a no-op because scheduling group membership is immutable in Kubernetes workload APIs.
func (ht *hierarchyTrackerImpl) OnPodUpdate(oldPod, newPod *v1.Pod) {
}

// OnPodDelete removes a pod from its scheduling group's active set.
// When removing a pod causes its PodGroup to drop below its minCount quorum, we propagate -1 readiness to the parent group.
func (ht *hierarchyTrackerImpl) OnPodDelete(pod *v1.Pod) {
	if pod == nil || pod.Spec.SchedulingGroup == nil || pod.Spec.SchedulingGroup.PodGroupName == nil {
		return
	}
	key := fwk.PodGroupKey(pod.Namespace, *pod.Spec.SchedulingGroup.PodGroupName)
	ht.lock.Lock()
	defer ht.lock.Unlock()

	group, exists := ht.groups[key]
	if !exists {
		return
	}
	uid := podUID(pod)
	if !group.activePods.Has(uid) {
		return
	}
	wasReady := ht.isGroupReady(group)
	group.activePods.Delete(uid)
	if wasReady && !ht.isGroupReady(group) {
		ht.propagateReadinessDelta(group.parentKey, -1)
	}
}

// updateGroupReadiness reconciles a group's minCount quorum and parent linkage. It assumes ht.lock is held.
// Because child pods or child groups may arrive before their parent PodGroup/CompositePodGroup definition,
// this method establishes the initial parent link and reconciles readiness transitions, propagating readiness deltas upwards.
func (ht *hierarchyTrackerImpl) updateGroupReadiness(key fwk.EntityKey, parentKey *fwk.EntityKey, minCount int) {
	group := ht.getOrCreateGroup(key)
	oldParent := group.parentKey
	wasReady := ht.isGroupReady(group)

	group.minCount = minCount
	group.parentKey = parentKey

	isNowReady := ht.isGroupReady(group)

	// If the parent link was newly resolved (e.g. definition arrived after child pods/groups),
	// retract readiness from the old parent (if any) and propagate to the new parent.
	if !ptr.Equal(oldParent, parentKey) {
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

// OnPodGroupAdd registers or updates a PodGroup in the hierarchy tracker and updates parent readiness if it is ready.
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
	ht.lock.Lock()
	defer ht.lock.Unlock()
	ht.updateGroupReadiness(key, parentKey, minCount)
}

// OnPodGroupUpdate re-evaluates PodGroup readiness and updates parent readiness when its policy changes.
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
	ht.lock.Lock()
	defer ht.lock.Unlock()

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

// OnCompositePodGroupAdd registers or updates a CompositePodGroup in the hierarchy tracker and updates parent readiness if it is ready.
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
	ht.lock.Lock()
	defer ht.lock.Unlock()
	ht.updateGroupReadiness(key, parentKey, minGroupCount)
}

// OnCompositePodGroupUpdate re-evaluates CompositePodGroup readiness and updates parent readiness when its policy changes.
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
	ht.lock.Lock()
	defer ht.lock.Unlock()

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
