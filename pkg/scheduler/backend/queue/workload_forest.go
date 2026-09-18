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

// workloadForest maintains a consistent view of observed GenericPodGroup objects (either PodGroup or CompositePodGroup).
// It ensures that scheduling queue invariants are preserved, independent of
// asynchronous updates happening in the scheduler cache.
// Outside of the scheduling queue, cache should be used as the source of truth.
// This structure is not thread-safe and should be accessed only under the lock of the PriorityQueue.
type workloadForest struct {
	podGroups map[fwk.EntityKey]*fwk.GenericPodGroup
	// children maps a parent CompositePodGroup key to its direct children keys (PodGroups or CompositePodGroups).
	// As an invariant of this structure, a child-to-parent relationship is populated here regardless of whether
	// the parent object has been explicitly observed yet. This prevents the need to iterate over all existing
	// groups to retroactively link children when a parent is finally added.
	children                   map[fwk.EntityKey]sets.Set[fwk.EntityKey]
	isCompositePodGroupEnabled bool
}

func newWorkloadForest(isCompositePodGroupEnabled bool) *workloadForest {
	return &workloadForest{
		podGroups:                  make(map[fwk.EntityKey]*fwk.GenericPodGroup),
		children:                   make(map[fwk.EntityKey]sets.Set[fwk.EntityKey]),
		isCompositePodGroupEnabled: isCompositePodGroupEnabled,
	}
}

// addGenericPodGroup adds a GenericPodGroup to the forest.
func (wf *workloadForest) addGenericPodGroup(gpg *fwk.GenericPodGroup) {
	key := gpg.GetKey()
	wf.podGroups[key] = gpg

	if !wf.isCompositePodGroupEnabled {
		return
	}
	parentKey, hasParent := gpg.GetParentKey()
	if !hasParent {
		return
	}

	_, exists := wf.children[parentKey]
	if !exists {
		wf.children[parentKey] = sets.New[fwk.EntityKey]()
	}
	wf.children[parentKey].Insert(key)
}

// updateGenericPodGroup updates a GenericPodGroup in the forest.
func (wf *workloadForest) updateGenericPodGroup(gpg *fwk.GenericPodGroup) {
	wf.podGroups[gpg.GetKey()] = gpg
}

// deleteGenericPodGroup removes a GenericPodGroup from the forest.
func (wf *workloadForest) deleteGenericPodGroup(gpg *fwk.GenericPodGroup) {
	key := gpg.GetKey()
	delete(wf.podGroups, key)

	if !wf.isCompositePodGroupEnabled {
		return
	}
	parentKey, hasParent := gpg.GetParentKey()
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

// getRootLookupInfo returns the lookup info of the current root PodGroup or CompositePodGroup for a given GenericPodGroup.
func (wf *workloadForest) getRootLookupInfo(gpg *fwk.GenericPodGroup) (*framework.QueuedPodGroupInfo, bool) {
	storedGPG, exists := wf.podGroups[gpg.GetKey()]
	if !exists {
		return nil, false
	}

	if !wf.isCompositePodGroupEnabled || !storedGPG.HasParent() {
		return &framework.QueuedPodGroupInfo{
			PodGroupInfo: &framework.PodGroupInfo{
				GenericPodGroup: storedGPG,
			},
		}, true
	}
	return wf.getRootLookupInfoForParentCPG(*storedGPG.GetParentCompositePodGroupName(), storedGPG.GetNamespace())
}

// getRootLookupInfoForParentCPG is a helper to traverse up the parent chain and return the lookup info of the root CompositePodGroup.
// It should be called only when the CompositePodGroup feature gate is enabled.
func (wf *workloadForest) getRootLookupInfoForParentCPG(parentName, namespace string) (*framework.QueuedPodGroupInfo, bool) {
	currParentName := parentName
	for range schedulingv1alpha3.WorkloadMaxTreeDepth - 1 {
		cpgKey := fwk.CompositePodGroupKey(namespace, currParentName)
		cpg, exists := wf.podGroups[cpgKey]
		if !exists {
			return nil, false
		}

		if !cpg.HasParent() {
			return newCompositePodGroupInfoForLookup(cpg.GetNamespace(), cpg.GetName()), true
		}
		currParentName = *cpg.GetParentCompositePodGroupName()
	}
	utilruntime.HandleError(fmt.Errorf("hierarchy exceeded maximum tree depth when getting root info: %s/%s", parentName, namespace))
	return nil, false
}

// getLeafPodGroups returns all PodGroups that are leaf nodes in the subtree rooted at the given rootLookupInfo.
func (wf *workloadForest) getLeafPodGroups(logger klog.Logger, rootLookupInfo *framework.QueuedPodGroupInfo) []*schedulingv1beta1.PodGroup {
	key := rootLookupInfo.GetKey()
	if rootLookupInfo.GetType() == fwk.PodGroupKeyType {
		gpg, exists := wf.podGroups[key]
		if !exists {
			return nil
		}
		return []*schedulingv1beta1.PodGroup{gpg.PodGroup}
	}

	var pgs []*schedulingv1beta1.PodGroup
	var collectLeaves func(currKey fwk.EntityKey, depth int) bool
	collectLeaves = func(currKey fwk.EntityKey, depth int) bool {
		if depth >= schedulingv1alpha3.WorkloadMaxTreeDepth {
			utilruntime.HandleErrorWithLogger(logger, nil, "Hierarchy exceeded maximum tree depth when getting leaf PodGroups", "compositePodGroup", klog.KObj(rootLookupInfo))
			return false
		}

		children, exists := wf.children[currKey]
		if !exists {
			return true
		}

		for childKey := range children {
			gpg, ok := wf.podGroups[childKey]
			if !ok {
				continue
			}
			if gpg.PodGroup != nil {
				pgs = append(pgs, gpg.PodGroup)
			} else if gpg.CompositePodGroup != nil {
				if !collectLeaves(childKey, depth+1) {
					return false
				}
			}
		}
		return true
	}
	if !collectLeaves(key, 0) {
		return nil
	}
	return pgs
}

// buildPodGroupInfo recursively constructs a PodGroupInfo representation for a given GenericPodGroup
// and all its children up to WorkloadMaxTreeDepth.
func (wf *workloadForest) buildPodGroupInfo(logger klog.Logger, gpg *fwk.GenericPodGroup, depth int) *framework.PodGroupInfo {
	if depth >= schedulingv1alpha3.WorkloadMaxTreeDepth {
		utilruntime.HandleErrorWithLogger(logger, nil, "Hierarchy exceeded maximum tree depth when building PodGroupInfo", "groupType", gpg.GetType(), "group", klog.KObj(gpg))
		return nil
	}

	pgi := &framework.PodGroupInfo{
		GenericPodGroup: gpg,
		Children:        make([]*framework.PodGroupInfo, 0),
	}

	key := gpg.GetKey()
	childrenSet, ok := wf.children[key]
	if !ok {
		return pgi
	}
	for childKey := range childrenSet {
		if childGPG, ok := wf.podGroups[childKey]; ok {
			childInfo := wf.buildPodGroupInfo(logger, childGPG, depth+1)
			if childInfo == nil {
				return nil
			}
			pgi.Children = append(pgi.Children, childInfo)
		}
	}
	return pgi
}

// buildQueuedPodGroupInfo constructs a QueuedPodGroupInfo starting from the provided root lookup info,
// building out the full hierarchy of PodGroupInfo nodes.
func (wf *workloadForest) buildQueuedPodGroupInfo(logger klog.Logger, rootLookup *framework.QueuedPodGroupInfo) *framework.QueuedPodGroupInfo {
	key := rootLookup.GetKey()
	gpg, ok := wf.podGroups[key]
	if !ok {
		return nil
	}
	pgi := wf.buildPodGroupInfo(logger, gpg, 0)
	if pgi == nil {
		return nil
	}
	return &framework.QueuedPodGroupInfo{
		PodGroupInfo: pgi,
	}
}
