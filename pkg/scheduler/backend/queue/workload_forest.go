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
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
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
	podGroups          map[fwk.EntityKey]*schedulingv1beta1.PodGroup
	compositePodGroups map[fwk.EntityKey]*schedulingv1alpha3.CompositePodGroup
	// children maps a parent CompositePodGroup key to its direct children keys (PodGroups or CompositePodGroups).
	// As an invariant of this structure, a child-to-parent relationship is populated here regardless of whether
	// the parent object has been explicitly observed yet. This prevents the need to iterate over all existing
	// groups to retroactively link children when a parent is finally added.
	children                   map[fwk.EntityKey]sets.Set[fwk.EntityKey]
	isCompositePodGroupEnabled bool
}

func newWorkloadForest(isCompositePodGroupEnabled bool) *workloadForest {
	return &workloadForest{
		podGroups:                  make(map[fwk.EntityKey]*schedulingv1beta1.PodGroup),
		compositePodGroups:         make(map[fwk.EntityKey]*schedulingv1alpha3.CompositePodGroup),
		children:                   make(map[fwk.EntityKey]sets.Set[fwk.EntityKey]),
		isCompositePodGroupEnabled: isCompositePodGroupEnabled,
	}
}

// addPodGroup adds a PodGroup to the forest.
func (wf *workloadForest) addPodGroup(podGroup *schedulingv1beta1.PodGroup) {
	pgKey := podGroupKey(podGroup)
	wf.podGroups[pgKey] = podGroup

	if !wf.podGroupHasParent(podGroup) {
		return
	}

	parentKey := fwk.CompositePodGroupKey(podGroup.Namespace, *podGroup.Spec.ParentCompositePodGroupName)
	_, exists := wf.children[parentKey]
	if !exists {
		wf.children[parentKey] = sets.New[fwk.EntityKey]()
	}
	wf.children[parentKey].Insert(pgKey)
}

// updatePodGroup updates a PodGroup in the forest.
func (wf *workloadForest) updatePodGroup(podGroup *schedulingv1beta1.PodGroup) {
	wf.podGroups[podGroupKey(podGroup)] = podGroup
}

// deletePodGroup removes a PodGroup from the forest.
func (wf *workloadForest) deletePodGroup(podGroup *schedulingv1beta1.PodGroup) {
	pgKey := podGroupKey(podGroup)
	delete(wf.podGroups, pgKey)

	if !wf.podGroupHasParent(podGroup) {
		return
	}

	parentKey := fwk.CompositePodGroupKey(podGroup.Namespace, *podGroup.Spec.ParentCompositePodGroupName)
	parentChildren, exists := wf.children[parentKey]
	if !exists {
		return
	}
	parentChildren.Delete(pgKey)
	if parentChildren.Len() == 0 {
		delete(wf.children, parentKey)
	}
}

// addCompositePodGroup adds a CompositePodGroup to the forest.
func (wf *workloadForest) addCompositePodGroup(cpg *schedulingv1alpha3.CompositePodGroup) {
	cpgKey := compositePodGroupKey(cpg)
	wf.compositePodGroups[cpgKey] = cpg

	if cpg.Spec.ParentCompositePodGroupName == nil {
		return
	}

	parentKey := fwk.CompositePodGroupKey(cpg.Namespace, *cpg.Spec.ParentCompositePodGroupName)
	_, exists := wf.children[parentKey]
	if !exists {
		wf.children[parentKey] = sets.New[fwk.EntityKey]()
	}
	wf.children[parentKey].Insert(cpgKey)
}

// updateCompositePodGroup updates a CompositePodGroup in the forest.
func (wf *workloadForest) updateCompositePodGroup(cpg *schedulingv1alpha3.CompositePodGroup) {
	wf.compositePodGroups[compositePodGroupKey(cpg)] = cpg
}

// deleteCompositePodGroup removes a CompositePodGroup from the forest.
func (wf *workloadForest) deleteCompositePodGroup(cpg *schedulingv1alpha3.CompositePodGroup) {
	cpgKey := compositePodGroupKey(cpg)
	delete(wf.compositePodGroups, cpgKey)

	if cpg.Spec.ParentCompositePodGroupName == nil {
		return
	}

	parentKey := fwk.CompositePodGroupKey(cpg.Namespace, *cpg.Spec.ParentCompositePodGroupName)
	parentChildren, exists := wf.children[parentKey]
	if !exists {
		return
	}

	parentChildren.Delete(cpgKey)
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
	return wf.getRootLookupInfoForPodGroup(podGroup)
}

// getRootLookupInfoForPodGroup returns the lookup info of the current root PodGroup or CompositePodGroup for a given PodGroup.
func (wf *workloadForest) getRootLookupInfoForPodGroup(podGroup *schedulingv1beta1.PodGroup) (*framework.QueuedPodGroupInfo, bool) {
	podGroup, exists := wf.podGroups[podGroupKey(podGroup)]
	if !exists {
		return nil, false
	}

	if !wf.podGroupHasParent(podGroup) {
		return &framework.QueuedPodGroupInfo{
			PodGroupInfo: &framework.PodGroupInfo{
				Namespace: podGroup.Namespace,
				Name:      podGroup.Name,
				Type:      fwk.PodGroupKeyType,
			},
		}, true
	}
	return wf.getRootLookupInfoForParentCPG(*podGroup.Spec.ParentCompositePodGroupName, podGroup.Namespace)
}

// getRootLookupInfoForCPG returns the lookup info of the current root CompositePodGroup for a given CompositePodGroup.
func (wf *workloadForest) getRootLookupInfoForCPG(cpg *schedulingv1alpha3.CompositePodGroup) (*framework.QueuedPodGroupInfo, bool) {
	return wf.getRootLookupInfoForParentCPG(cpg.Name, cpg.Namespace)
}

// getRootLookupInfoForParentCPG is a helper to traverse up the parent chain and return the lookup info of the root CompositePodGroup.
func (wf *workloadForest) getRootLookupInfoForParentCPG(parentName, namespace string) (*framework.QueuedPodGroupInfo, bool) {
	currParentName := parentName
	visited := sets.New[fwk.EntityKey]()
	for {
		cpgKey := fwk.CompositePodGroupKey(namespace, currParentName)
		if visited.Has(cpgKey) {
			// TODO(jdzikowski): propagate logger to the getPod method in the scheduling queue.
			utilruntime.HandleError(fmt.Errorf("cycle detected in composite pod group hierarchy: %s/%s", parentName, namespace))
			return nil, false
		}
		visited.Insert(cpgKey)

		cpg, exists := wf.compositePodGroups[cpgKey]
		if !exists {
			return nil, false
		}

		if cpg.Spec.ParentCompositePodGroupName == nil {
			return &framework.QueuedPodGroupInfo{
				PodGroupInfo: &framework.PodGroupInfo{
					Namespace: cpg.Namespace,
					Name:      cpg.Name,
					Type:      fwk.CompositePodGroupKeyType,
				},
			}, true
		}
		currParentName = *cpg.Spec.ParentCompositePodGroupName
	}
}

// getLeafPodGroups returns all PodGroups that are leaf nodes in the subtree rooted at the given CompositePodGroup.
func (wf *workloadForest) getLeafPodGroups(logger klog.Logger, rootLookupInfo *framework.QueuedPodGroupInfo) []*schedulingv1beta1.PodGroup {
	var pgs []*schedulingv1beta1.PodGroup
	var key fwk.EntityKey
	if rootLookupInfo.GetType() == fwk.PodGroupKeyType {
		key = fwk.PodGroupKey(rootLookupInfo.GetNamespace(), rootLookupInfo.GetName())
		pg, exists := wf.podGroups[key]
		if !exists {
			return pgs
		}
		pgs = append(pgs, pg)
		return pgs
	}

	key = fwk.CompositePodGroupKey(rootLookupInfo.GetNamespace(), rootLookupInfo.GetName())
	queue := []fwk.EntityKey{key}
	visited := sets.New[fwk.EntityKey]()

	for len(queue) > 0 {
		currKey := queue[0]
		queue = queue[1:]

		if visited.Has(currKey) {
			utilruntime.HandleErrorWithLogger(logger, nil, "cycle detected in composite pod group hierarchy", "cpg-name", rootLookupInfo.GetName(), "namespace", rootLookupInfo.GetNamespace())
			return pgs
		}
		visited.Insert(currKey)

		children, exists := wf.children[currKey]
		if !exists {
			continue
		}

		for childKey := range children {
			if pg, isPG := wf.podGroups[childKey]; isPG {
				pgs = append(pgs, pg)
			}
			if _, isCPG := wf.compositePodGroups[childKey]; isCPG {
				queue = append(queue, childKey)
			}
		}
	}

	return pgs
}

// getPodGroup returns the current PodGroup object for a given lookup.
func (wf *workloadForest) getPodGroup(pgLookup *schedulingv1beta1.PodGroup) (*schedulingv1beta1.PodGroup, bool) {
	podGroup, ok := wf.podGroups[podGroupKey(pgLookup)]
	return podGroup, ok
}

// getCompositePodGroup returns the current CompositePodGroup object for a given lookup.
func (wf *workloadForest) getCompositePodGroup(cpgLookup *schedulingv1alpha3.CompositePodGroup) (*schedulingv1alpha3.CompositePodGroup, bool) {
	podGroup, ok := wf.compositePodGroups[compositePodGroupKey(cpgLookup)]
	return podGroup, ok
}

// buildPodGroupInfoForPG constructs a PodGroupInfo representation for a given PodGroup,
// using the provided visited set to detect cycles in the hierarchy.
func (wf *workloadForest) buildPodGroupInfoForPG(logger klog.Logger, pg *schedulingv1beta1.PodGroup, visited sets.Set[fwk.EntityKey]) *framework.PodGroupInfo {
	key := podGroupKey(pg)
	if visited.Has(key) {
		utilruntime.HandleErrorWithLogger(logger, nil, "cycle detected in composite pod group hierarchy", "pg-name", pg.Name, "namespace", pg.Namespace)
		return nil
	}
	visited.Insert(key)

	return &framework.PodGroupInfo{
		Namespace: pg.Namespace,
		Name:      pg.Name,
		Type:      fwk.PodGroupKeyType,
		PodGroup:  pg,
		Children:  make([]*framework.PodGroupInfo, 0),
	}
}

// buildPodGroupInfoForCPG recursively constructs a PodGroupInfo representation for a given CompositePodGroup
// and all its children, using the provided visited set to detect cycles in the hierarchy.
func (wf *workloadForest) buildPodGroupInfoForCPG(logger klog.Logger, cpg *schedulingv1alpha3.CompositePodGroup, visited sets.Set[fwk.EntityKey]) *framework.PodGroupInfo {
	key := compositePodGroupKey(cpg)
	if visited.Has(key) {
		utilruntime.HandleErrorWithLogger(logger, nil, "cycle detected in composite pod group hierarchy", "cpg-name", cpg.Name, "namespace", cpg.Namespace)
		return nil
	}
	visited.Insert(key)

	pgi := &framework.PodGroupInfo{
		Namespace:         cpg.Namespace,
		Name:              cpg.Name,
		Type:              fwk.CompositePodGroupKeyType,
		CompositePodGroup: cpg,
		Children:          make([]*framework.PodGroupInfo, 0),
	}

	childrenSet, ok := wf.children[key]
	if !ok {
		return pgi
	}
	for childKey := range childrenSet {
		if childPG, ok := wf.podGroups[childKey]; ok {
			if childInfo := wf.buildPodGroupInfoForPG(logger, childPG, visited); childInfo != nil {
				pgi.Children = append(pgi.Children, childInfo)
			}
		}
		if childCPG, ok := wf.compositePodGroups[childKey]; ok {
			if childInfo := wf.buildPodGroupInfoForCPG(logger, childCPG, visited); childInfo != nil {
				pgi.Children = append(pgi.Children, childInfo)
			}
		}
	}
	return pgi
}

// buildQueuedPodGroupInfo constructs a QueuedPodGroupInfo starting from the provided root lookup info,
// building out the full hierarchy of PodGroupInfo nodes and initializing the QueuedPodInfos map.
func (wf *workloadForest) buildQueuedPodGroupInfo(logger klog.Logger, rootLookup *framework.QueuedPodGroupInfo) *framework.QueuedPodGroupInfo {
	var pgi *framework.PodGroupInfo
	switch rootLookup.GetType() {
	case fwk.PodGroupKeyType:
		if pg, ok := wf.podGroups[fwk.PodGroupKey(rootLookup.GetNamespace(), rootLookup.GetName())]; ok && pg != nil {
			pgi = wf.buildPodGroupInfoForPG(logger, pg, sets.New[fwk.EntityKey]())
		}
	case fwk.CompositePodGroupKeyType:
		if cpg, ok := wf.compositePodGroups[fwk.CompositePodGroupKey(rootLookup.GetNamespace(), rootLookup.GetName())]; ok && cpg != nil {
			pgi = wf.buildPodGroupInfoForCPG(logger, cpg, sets.New[fwk.EntityKey]())
		}
	}

	if pgi == nil {
		return nil
	}
	return &framework.QueuedPodGroupInfo{
		PodGroupInfo:   pgi,
		QueuedPodInfos: make(map[fwk.EntityKey][]*framework.QueuedPodInfo),
	}
}

// podGroupHasParent checks whether the given PodGroup has a parent CompositePodGroup configured
// and the CompositePodGroup feature is enabled.
func (wf *workloadForest) podGroupHasParent(pg *schedulingv1beta1.PodGroup) bool {
	return wf.isCompositePodGroupEnabled && pg.Spec.ParentCompositePodGroupName != nil
}
