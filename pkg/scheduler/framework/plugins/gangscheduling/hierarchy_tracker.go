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
	"fmt"
	"slices"
	"sync"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/utils/ptr"
)

const cycleDetectedMsgFmt = "cycle detected in the hierarchy: %v"

// hierarchyGroup tracks the scheduling policy's readiness constraint (minCount / minGroupCount / basic policy) and
// state of a single PodGroup or CompositePodGroup. We store allPods for leaf PodGroups and readyChildren for
// CompositePodGroups so that readiness can be evaluated uniformly across different levels of the hierarchy.
type hierarchyGroup struct {
	key           fwk.EntityKey
	genericGroup  *fwk.GenericPodGroup
	parentKey     *fwk.EntityKey
	minCount      int
	allPods       sets.Set[types.UID]
	readyChildren int
}

// empty returns true when the API object definition is absent and no pods or ready children remain.
func (g *hierarchyGroup) empty() bool {
	if g.genericGroup != nil {
		return false
	}
	if g.key.Type == fwk.PodGroupKeyType {
		return len(g.allPods) == 0
	}
	return g.readyChildren == 0
}

// readyCount returns the number of pods for a PodGroup or ready child groups for a CompositePodGroup.
func (g *hierarchyGroup) readyCount() int {
	if g.key.Type == fwk.PodGroupKeyType {
		return len(g.allPods)
	}
	return g.readyChildren
}

// HierarchyTracker incrementally tracks readiness constraint counts across PodGroup and CompositePodGroup
// hierarchies. Its in-memory index is protected by an RWMutex to allow concurrent PreEnqueue lookups
// from scheduling worker routines.
type HierarchyTracker struct {
	lock                     sync.RWMutex
	compositePodGroupEnabled bool
	groups                   map[fwk.EntityKey]*hierarchyGroup
}

var _ fwk.PodGroupHierarchyTracker = &HierarchyTracker{}

// NewHierarchyTracker returns a new thread-safe HierarchyTracker.
func NewHierarchyTracker(compositePodGroupEnabled bool) *HierarchyTracker {
	return &HierarchyTracker{
		compositePodGroupEnabled: compositePodGroupEnabled,
		groups:                   make(map[fwk.EntityKey]*hierarchyGroup),
	}
}

// findRootGroup traverses parent links from key to locate the top-most root hierarchyGroup in the hierarchy.
// If key already identifies the root, the hierarchyGroup corresponding to key is returned.
// If the root group cannot be resolved (or does not exist), nil is returned without an error.
// Assumes ht.lock is held.
func (ht *HierarchyTracker) findRootGroup(key fwk.EntityKey) (*hierarchyGroup, error) {
	currentKey := key
	var visited []fwk.EntityKey
	for {
		if slices.Contains(visited, currentKey) {
			visited = append(visited, currentKey)
			return nil, fmt.Errorf(cycleDetectedMsgFmt, visited)
		}
		visited = append(visited, currentKey)

		group, exists := ht.groups[currentKey]
		if !exists || group.genericGroup == nil {
			return nil, nil
		}
		if !ht.compositePodGroupEnabled || group.parentKey == nil {
			return group, nil
		}
		currentKey = *group.parentKey
	}
}

// FindRootGroupReadiness traverses parent links from key to locate the root GenericPodGroup in the hierarchy.
// If the root group is found, it is returned along with information if the number of its children meets its scheduling policy's
// minimum threshold. If the root group cannot be resolved (or does not exist), nil is returned without an error.
func (ht *HierarchyTracker) FindRootGroupReadiness(key fwk.EntityKey) (*fwk.RootGroupReadiness, error) {
	ht.lock.RLock()
	defer ht.lock.RUnlock()

	rootGroup, err := ht.findRootGroup(key)
	if err != nil || rootGroup == nil {
		return nil, err
	}
	return &fwk.RootGroupReadiness{
		RootGroup: rootGroup.genericGroup,
		IsReady:   ht.isGroupReady(rootGroup),
	}, nil
}

// AreSameHierarchy checks whether two entities belong to the same hierarchy owned by the same root group.
func (ht *HierarchyTracker) AreSameHierarchy(key1, key2 fwk.EntityKey) (bool, error) {
	ht.lock.RLock()
	defer ht.lock.RUnlock()

	root1, err := ht.findRootGroup(key1)
	if err != nil || root1 == nil {
		return false, err
	}
	root2, err := ht.findRootGroup(key2)
	if err != nil || root2 == nil {
		return false, err
	}
	return root1.key == root2.key, nil
}

// getOrCreateGroup retrieves or initializes a hierarchyGroup. It assumes ht.lock is held.
// We default minCount to 1 so that out-of-order informer events (e.g. a Pod arriving before its PodGroup)
// still enforce a baseline quorum until the authoritative policy definition arrives.
func (ht *HierarchyTracker) getOrCreateGroup(key fwk.EntityKey) *hierarchyGroup {
	group, exists := ht.groups[key]
	if !exists {
		group = &hierarchyGroup{
			key:      key,
			minCount: 1,
			allPods:  sets.New[types.UID](),
		}
		ht.groups[key] = group
	}
	return group
}

// isGroupReady reports whether the group currently satisfies its readiness requirement. It assumes ht.lock is held.
// Leaf PodGroups evaluate readiness against active pods; CompositePodGroups evaluate against ready child groups.
func (ht *HierarchyTracker) isGroupReady(group *hierarchyGroup) bool {
	if group == nil {
		return false
	}
	if group.key.Type == fwk.PodGroupKeyType {
		return group.allPods.Len() >= group.minCount
	}
	return group.readyChildren >= group.minCount
}

// propagateReadinessDelta applies a readiness delta up the group hierarchy. It assumes ht.lock is held.
// Upward propagation continues only while a parent transitions across its readiness boundary,
// preventing redundant updates when a group gains or loses members above or below its minCount quorum.
func (ht *HierarchyTracker) propagateReadinessDelta(logger klog.Logger, parentKey *fwk.EntityKey, delta int) {
	var visited []fwk.EntityKey
	for parentKey != nil && delta != 0 {
		if slices.Contains(visited, *parentKey) {
			visited = append(visited, *parentKey)
			utilruntime.HandleErrorWithLogger(logger, nil, fmt.Sprintf(cycleDetectedMsgFmt, visited))
			return
		}
		visited = append(visited, *parentKey)

		parent := ht.getOrCreateGroup(*parentKey)
		wasReady := ht.isGroupReady(parent)
		parent.readyChildren += delta
		isNowReady := ht.isGroupReady(parent)

		nextKey, nextDelta := parent.parentKey, 0
		if wasReady != isNowReady {
			nextDelta = -1
			if isNowReady {
				nextDelta = 1
			}
		}
		if delta < 0 && parent.empty() {
			delete(ht.groups, *parentKey)
		}

		parentKey, delta = nextKey, nextDelta
	}
}

// AddPod records a pod as active in its scheduling group.
// When adding a pod causes its PodGroup to cross its minCount quorum, we propagate +1 readiness to the parent group.
func (ht *HierarchyTracker) AddPod(logger klog.Logger, pod *v1.Pod) {
	if pod.Spec.SchedulingGroup == nil || pod.Spec.SchedulingGroup.PodGroupName == nil {
		return
	}

	ht.lock.Lock()
	defer ht.lock.Unlock()
	ht.addPod(logger, pod)
}

// addPod records a pod as active in its scheduling group. It assumes ht.lock is held.
func (ht *HierarchyTracker) addPod(logger klog.Logger, pod *v1.Pod) {
	key := fwk.PodGroupKey(pod.Namespace, *pod.Spec.SchedulingGroup.PodGroupName)
	group := ht.getOrCreateGroup(key)
	if group.allPods.Has(pod.UID) {
		return
	}
	wasReady := ht.isGroupReady(group)
	group.allPods.Insert(pod.UID)
	if !wasReady && ht.isGroupReady(group) {
		ht.propagateReadinessDelta(logger, group.parentKey, 1)
	}
}

// UpdatePod handles a pod recreated under the same name, which an informer can collapse into a single update.
func (ht *HierarchyTracker) UpdatePod(logger klog.Logger, oldPod, newPod *v1.Pod) {
	if oldPod.UID == newPod.UID {
		return
	}
	skipDelete := oldPod.Spec.SchedulingGroup == nil || oldPod.Spec.SchedulingGroup.PodGroupName == nil
	skipAdd := newPod.Spec.SchedulingGroup == nil || newPod.Spec.SchedulingGroup.PodGroupName == nil
	if skipDelete && skipAdd {
		return
	}

	ht.lock.Lock()
	defer ht.lock.Unlock()
	if !skipDelete {
		ht.deletePod(logger, oldPod)
	}
	if !skipAdd {
		ht.addPod(logger, newPod)
	}
}

// DeletePod removes a pod from its scheduling group's active set.
// When removing a pod causes its PodGroup to drop below its minCount quorum, we propagate -1 readiness to the parent group.
func (ht *HierarchyTracker) DeletePod(logger klog.Logger, pod *v1.Pod) {
	if pod.Spec.SchedulingGroup == nil || pod.Spec.SchedulingGroup.PodGroupName == nil {
		return
	}
	ht.lock.Lock()
	defer ht.lock.Unlock()
	ht.deletePod(logger, pod)
}

// deletePod removes a pod from its scheduling group's active set. It assumes ht.lock is held.
func (ht *HierarchyTracker) deletePod(logger klog.Logger, pod *v1.Pod) {
	key := fwk.PodGroupKey(pod.Namespace, *pod.Spec.SchedulingGroup.PodGroupName)
	group, exists := ht.groups[key]
	if !exists {
		return
	}
	if !group.allPods.Has(pod.UID) {
		return
	}
	wasReady := ht.isGroupReady(group)
	group.allPods.Delete(pod.UID)
	if wasReady && !ht.isGroupReady(group) {
		ht.propagateReadinessDelta(logger, group.parentKey, -1)
	}
	if group.empty() {
		delete(ht.groups, key)
	}
}

// updateGroupReadiness reconciles a group's minCount quorum and parent linkage. It assumes ht.lock is held.
// Because child pods or child groups may arrive before their parent PodGroup/CompositePodGroup definition,
// this method establishes the initial parent link and reconciles readiness transitions, propagating readiness deltas upwards.
func (ht *HierarchyTracker) updateGroupReadiness(logger klog.Logger, group *hierarchyGroup, parentKey *fwk.EntityKey, minCount int) {
	oldParent := group.parentKey
	wasReady := ht.isGroupReady(group)

	group.minCount = minCount
	group.parentKey = parentKey

	isNowReady := ht.isGroupReady(group)

	// Parent link can change when resolved from nil (out-of-order arrival) or when an informer
	// collapses a delete+recreate into an Update. Retract readiness from old parent and propagate to new.
	if !ptr.Equal(oldParent, parentKey) {
		if wasReady {
			ht.propagateReadinessDelta(logger, oldParent, -1)
		}
		if isNowReady {
			ht.propagateReadinessDelta(logger, parentKey, 1)
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
	ht.propagateReadinessDelta(logger, parentKey, delta)
}

// AddGenericPodGroup registers or updates a GenericPodGroup in the hierarchy tracker and updates parent readiness if it is ready.
func (ht *HierarchyTracker) AddGenericPodGroup(logger klog.Logger, gpg *fwk.GenericPodGroup) {
	key := gpg.GetKey()
	minCount := gpg.GetMinCount()
	var parentKey *fwk.EntityKey
	if ht.compositePodGroupEnabled {
		if pKey, hasParent := gpg.GetParentKey(); hasParent {
			parentKey = &pKey
		}
	}

	ht.lock.Lock()
	defer ht.lock.Unlock()

	group := ht.getOrCreateGroup(key)
	group.genericGroup = gpg
	ht.updateGroupReadiness(logger, group, parentKey, minCount)
}

// UpdateGenericPodGroup re-evaluates GenericPodGroup readiness and parent linkage when its definition changes.
func (ht *HierarchyTracker) UpdateGenericPodGroup(logger klog.Logger, gpg *fwk.GenericPodGroup) {
	ht.AddGenericPodGroup(logger, gpg)
}

// DeleteGenericPodGroup removes a GenericPodGroup's definition from tracking.
// If the group was previously ready, we retract its readiness from its parent before removing.
func (ht *HierarchyTracker) DeleteGenericPodGroup(logger klog.Logger, gpg *fwk.GenericPodGroup) {
	key := gpg.GetKey()
	ht.lock.Lock()
	defer ht.lock.Unlock()

	group, exists := ht.groups[key]
	if !exists {
		return
	}
	wasReady := ht.isGroupReady(group)
	group.genericGroup = nil
	if wasReady {
		ht.propagateReadinessDelta(logger, group.parentKey, -1)
	}
	group.parentKey = nil
	if group.empty() {
		delete(ht.groups, key)
	}
}
