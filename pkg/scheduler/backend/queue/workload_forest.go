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

package queue

import (
	"fmt"

	v1 "k8s.io/api/core/v1"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

// workloadForest maintains a consistent view of observed PodGroup objects.
// It ensures that scheduling queue invariants are preserved, independent of
// asynchronous updates happening in the scheduler cache.
// Outside of the scheduling queue, cache should be used as the source of truth.
// This structure is not thread-safe and should be accessed only under the lock of the PriorityQueue.
type workloadForest struct {
	podGroups map[fwk.EntityKey]*framework.AbstractPodGroup
	// children maps a parent CompositePodGroup key to its direct children keys (PodGroups or CompositePodGroups).
	// As an invariant of this structure, a child-to-parent relationship is populated here regardless of whether
	// the parent object has been explicitly observed yet. This prevents the need to iterate over all existing
	// groups to retroactively link children when a parent is finally added.
	children                   map[fwk.EntityKey]sets.Set[fwk.EntityKey]
	isCompositePodGroupEnabled bool
}

func newWorkloadForest(isCompositePodGroupEnabled bool) *workloadForest {
	return &workloadForest{
		podGroups:                  make(map[fwk.EntityKey]*framework.AbstractPodGroup),
		children:                   make(map[fwk.EntityKey]sets.Set[fwk.EntityKey]),
		isCompositePodGroupEnabled: isCompositePodGroupEnabled,
	}
}

// addAbstractPodGroup adds an AbstractPodGroup to the forest.
func (wf *workloadForest) addAbstractPodGroup(apg *framework.AbstractPodGroup) {
	key := apg.GetKey()
	wf.podGroups[key] = apg

	if !wf.isCompositePodGroupEnabled {
		return
	}
	parentKey, hasParent := apg.GetParentKey()
	if !hasParent {
		return
	}

	_, exists := wf.children[parentKey]
	if !exists {
		wf.children[parentKey] = sets.New[fwk.EntityKey]()
	}
	wf.children[parentKey].Insert(key)
}

// updateAbstractPodGroup updates an AbstractPodGroup in the forest.
func (wf *workloadForest) updateAbstractPodGroup(apg *framework.AbstractPodGroup) {
	wf.podGroups[apg.GetKey()] = apg
}

// deleteAbstractPodGroup removes an AbstractPodGroup from the forest.
func (wf *workloadForest) deleteAbstractPodGroup(apg *framework.AbstractPodGroup) {
	key := apg.GetKey()
	delete(wf.podGroups, key)

	if !wf.isCompositePodGroupEnabled {
		return
	}
	parentKey, hasParent := apg.GetParentKey()
	if !hasParent {
		return
	}

	parentChildren, exists := wf.children[parentKey]
	if !exists {
		return
	}
	parentChildren.Delete(key)
	if parentChildren.Len() == 0 {
		delete(wf.children, parentKey)
	}
}

// getRootLookupInfoForPod returns the lookup info of the current root PodGroup or CompositePodGroup for a given pod.
func (wf *workloadForest) getRootLookupInfoForPod(pod *v1.Pod) (*framework.QueuedPodGroupInfo, bool) {
	podGroup, exists := wf.podGroups[podGroupKeyForPod(pod)]
	if !exists {
		return nil, false
	}
	return wf.getRootLookupInfo(podGroup)
}

// getRootLookupInfo returns the lookup info of the current root PodGroup or CompositePodGroup for a given AbstractPodGroup.
func (wf *workloadForest) getRootLookupInfo(apg *framework.AbstractPodGroup) (*framework.QueuedPodGroupInfo, bool) {
	storedAPG, exists := wf.podGroups[apg.GetKey()]
	if !exists {
		return nil, false
	}

	if !wf.isCompositePodGroupEnabled || !storedAPG.HasParent() {
		return &framework.QueuedPodGroupInfo{
			PodGroupInfo: &framework.PodGroupInfo{
				Namespace: storedAPG.GetNamespace(),
				Name:      storedAPG.GetName(),
				Type:      storedAPG.GetType(),
			},
		}, true
	}
	return wf.getRootLookupInfoForParentCPG(*storedAPG.GetParentCompositePodGroupName(), storedAPG.GetNamespace())
}

// getRootLookupInfoForParentCPG is a helper to traverse up the parent chain and return the lookup info of the root CompositePodGroup.
func (wf *workloadForest) getRootLookupInfoForParentCPG(parentName, namespace string) (*framework.QueuedPodGroupInfo, bool) {
	currParentName := parentName
	visited := sets.New[fwk.EntityKey]()
	for {
		cpgKey := fwk.CompositePodGroupKey(namespace, currParentName)
		if visited.Has(cpgKey) {
			// TODO(jdzikowski): propagate logger to the getPod method in the scheduling queue.
			utilruntime.HandleError(fmt.Errorf("cycle detected in composite pod group hierarchy when getting root info: %s/%s", parentName, namespace))
			return nil, false
		}
		visited.Insert(cpgKey)

		cpg, exists := wf.podGroups[cpgKey]
		if !exists {
			return nil, false
		}

		if !cpg.HasParent() {
			return &framework.QueuedPodGroupInfo{
				PodGroupInfo: &framework.PodGroupInfo{
					Namespace: cpg.GetNamespace(),
					Name:      cpg.GetName(),
					Type:      fwk.CompositePodGroupKeyType,
				},
			}, true
		}
		currParentName = *cpg.GetParentCompositePodGroupName()
	}
}

// getLeafPodGroups returns all PodGroups that are leaf nodes in the subtree rooted at the given CompositePodGroup.
func (wf *workloadForest) getLeafPodGroups(logger klog.Logger, rootLookupInfo *framework.QueuedPodGroupInfo) []*schedulingv1beta1.PodGroup {
	var key fwk.EntityKey
	if rootLookupInfo.GetType() == fwk.PodGroupKeyType {
		key = fwk.PodGroupKey(rootLookupInfo.GetNamespace(), rootLookupInfo.GetName())
		apg, exists := wf.podGroups[key]
		if !exists {
			return nil
		}
		return []*schedulingv1beta1.PodGroup{apg.PodGroup}
	}

	var pgs []*schedulingv1beta1.PodGroup
	key = fwk.CompositePodGroupKey(rootLookupInfo.GetNamespace(), rootLookupInfo.GetName())
	queue := []fwk.EntityKey{key}
	visited := sets.New[fwk.EntityKey]()

	for len(queue) > 0 {
		currKey := queue[0]
		queue = queue[1:]

		if visited.Has(currKey) {
			utilruntime.HandleErrorWithLogger(logger, nil, "Cycle detected in composite pod group hierarchy when getting leaf PodGroups", "compositePodGroup", klog.KObj(rootLookupInfo))
			return pgs
		}
		visited.Insert(currKey)

		children, exists := wf.children[currKey]
		if !exists {
			continue
		}

		for childKey := range children {
			apg, ok := wf.podGroups[childKey]
			if !ok {
				continue
			}
			if apg.PodGroup != nil {
				pgs = append(pgs, apg.PodGroup)
			} else if apg.CompositePodGroup != nil {
				queue = append(queue, childKey)
			}
		}
	}

	return pgs
}

// buildPodGroupInfo recursively constructs a PodGroupInfo representation for a given AbstractPodGroup
// and all its children, using the provided visited set to detect cycles in the hierarchy.
func (wf *workloadForest) buildPodGroupInfo(logger klog.Logger, apg *framework.AbstractPodGroup, visited sets.Set[fwk.EntityKey]) *framework.PodGroupInfo {
	key := apg.GetKey()
	if visited.Has(key) {
		utilruntime.HandleErrorWithLogger(logger, nil, "Cycle detected in composite pod group hierarchy when building PodGroupInfo", "groupType", apg.GetType(), "group", klog.KObj(apg))
		return nil
	}
	visited.Insert(key)

	pgi := &framework.PodGroupInfo{
		Namespace:         apg.GetNamespace(),
		Name:              apg.GetName(),
		Type:              apg.GetType(),
		PodGroup:          apg.PodGroup,
		CompositePodGroup: apg.CompositePodGroup,
		Children:          make([]*framework.PodGroupInfo, 0),
	}

	childrenSet, ok := wf.children[key]
	if !ok {
		return pgi
	}
	for childKey := range childrenSet {
		if childAPG, ok := wf.podGroups[childKey]; ok {
			if childInfo := wf.buildPodGroupInfo(logger, childAPG, visited); childInfo != nil {
				pgi.Children = append(pgi.Children, childInfo)
			}
		}
	}
	return pgi
}

// buildQueuedPodGroupInfo constructs a QueuedPodGroupInfo starting from the provided root lookup info,
// building out the full hierarchy of PodGroupInfo nodes and initializing the QueuedPodInfos map.
func (wf *workloadForest) buildQueuedPodGroupInfo(logger klog.Logger, rootLookup *framework.QueuedPodGroupInfo) *framework.QueuedPodGroupInfo {
	var key fwk.EntityKey
	switch rootLookup.GetType() {
	case fwk.PodGroupKeyType:
		key = fwk.PodGroupKey(rootLookup.GetNamespace(), rootLookup.GetName())
	case fwk.CompositePodGroupKeyType:
		key = fwk.CompositePodGroupKey(rootLookup.GetNamespace(), rootLookup.GetName())
	}

	apg, ok := wf.podGroups[key]
	if !ok {
		return nil
	}
	return &framework.QueuedPodGroupInfo{
		PodGroupInfo:   wf.buildPodGroupInfo(logger, apg, sets.New[fwk.EntityKey]()),
		QueuedPodInfos: make(map[fwk.EntityKey][]*framework.QueuedPodInfo),
	}
}
