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
	v1 "k8s.io/api/core/v1"
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	"k8s.io/apimachinery/pkg/util/sets"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

// workloadForest maintains a consistent view of observed PodGroup objects.
// It ensures that scheduling queue invariants are preserved, independent of
// asynchronous updates happening in the scheduler cache.
// Outside of the scheduling queue, cache should be used as the source of truth.
type workloadForest struct {
	podGroups          map[string]*schedulingv1alpha3.PodGroup
	compositePodGroups map[string]*schedulingv1alpha3.CompositePodGroup
	// Children can have entries for non-existent parent keys. We do it to improve
	// performance by avoiding iteration over all PodGroups and CompositePodGroups
	// while adding a parent CompositePodGroup.
	children map[string]sets.Set[string]
}

func newWorkloadForest() *workloadForest {
	return &workloadForest{
		podGroups:          make(map[string]*schedulingv1alpha3.PodGroup),
		compositePodGroups: make(map[string]*schedulingv1alpha3.CompositePodGroup),
		children:           make(map[string]sets.Set[string]),
	}
}

// addPodGroup adds a PodGroup to the forest.
func (wf *workloadForest) addPodGroup(podGroup *schedulingv1alpha3.PodGroup) {
	pgKey := podGroupKey(podGroup)
	wf.podGroups[pgKey] = podGroup

	if podGroup.Spec.ParentCompositePodGroupName == nil {
		return
	}

	parentKey := compositePodGroupKeyFromName(*podGroup.Spec.ParentCompositePodGroupName, podGroup.Namespace)
	_, exists := wf.children[parentKey]
	if !exists {
		wf.children[parentKey] = sets.New[string]()
	}
	wf.children[parentKey].Insert(pgKey)
}

// updatePodGroup updates a PodGroup in the forest.
func (wf *workloadForest) updatePodGroup(podGroup *schedulingv1alpha3.PodGroup) {
	wf.podGroups[podGroupKey(podGroup)] = podGroup
}

// deletePodGroup removes a PodGroup from the forest.
func (wf *workloadForest) deletePodGroup(podGroup *schedulingv1alpha3.PodGroup) {
	pgKey := podGroupKey(podGroup)
	delete(wf.podGroups, pgKey)

	if podGroup.Spec.ParentCompositePodGroupName == nil {
		return
	}

	parentKey := compositePodGroupKeyFromName(*podGroup.Spec.ParentCompositePodGroupName, podGroup.Namespace)
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
	if _, exists := wf.children[cpgKey]; !exists {
		wf.children[cpgKey] = sets.New[string]()
	}

	if cpg.Spec.ParentCompositePodGroupName == nil {
		return
	}

	parentKey := compositePodGroupKeyFromName(*cpg.Spec.ParentCompositePodGroupName, cpg.Namespace)
	_, exists := wf.children[parentKey]
	if !exists {
		wf.children[parentKey] = sets.New[string]()
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
	delete(wf.children, cpgKey)

	if cpg.Spec.ParentCompositePodGroupName == nil {
		return
	}

	parentKey := compositePodGroupKeyFromName(*cpg.Spec.ParentCompositePodGroupName, cpg.Namespace)
	parentChildren, exists := wf.children[parentKey]
	if !exists {
		return
	}

	parentChildren.Delete(cpgKey)
	if parentChildren.Len() == 0 {
		delete(wf.children, parentKey)
	}
}

// getRootForPod returns the current root PodGroup or CompositePodGroup object for a given pod.
// TODO (jakiś interace na any???)
// TODO (metoda pathtoroot i getrootforpod zwraca klucz a nie any??)
func (wf *workloadForest) getRootForPod(pod *v1.Pod) (any, bool) {
	podGroup, exists := wf.podGroups[podGroupKeyForPod(pod)]
	if !exists {
		return nil, false
	}

	if podGroup.Spec.ParentCompositePodGroupName == nil {
		return podGroup, true
	}

	currParentName := *podGroup.Spec.ParentCompositePodGroupName
	for {
		cpgKey := compositePodGroupKeyFromName(currParentName, podGroup.Namespace)
		cpg, exists := wf.compositePodGroups[cpgKey]
		if !exists {
			return nil, false
		}

		if cpg.Spec.ParentCompositePodGroupName == nil {
			return cpg, true
		}
		currParentName = *cpg.Spec.ParentCompositePodGroupName
	}
}

// getRootLookupInfoForPod returns the lookup info of the current root PodGroup or CompositePodGroup for a given pod.
func (wf *workloadForest) getRootLookupInfoForPod(pod *v1.Pod) (*framework.QueuedPodGroupInfo, bool) {
	podGroup, exists := wf.podGroups[podGroupKeyForPod(pod)]
	if !exists {
		return nil, false
	}

	if podGroup.Spec.ParentCompositePodGroupName == nil {
		return &framework.QueuedPodGroupInfo{
			PodGroupInfo: &framework.PodGroupInfo{
				Namespace: podGroup.Namespace,
				Name:      podGroup.Name,
				Type:      framework.PodGroupKeyType,
			},
		}, true
	}

	currParentName := *podGroup.Spec.ParentCompositePodGroupName
	for {
		cpgKey := compositePodGroupKeyFromName(currParentName, podGroup.Namespace)
		cpg, exists := wf.compositePodGroups[cpgKey]
		if !exists {
			return nil, false
		}

		if cpg.Spec.ParentCompositePodGroupName == nil {
			return &framework.QueuedPodGroupInfo{
				PodGroupInfo: &framework.PodGroupInfo{
					Namespace: cpg.Namespace,
					Name:      cpg.Name,
					Type:      framework.CompositePodGroupKeyType,
				},
			}, true
		}
		currParentName = *cpg.Spec.ParentCompositePodGroupName
	}
}

// getPodGroup returns the current PodGroup object for a given lookup.
func (wf *workloadForest) getPodGroup(pgLookup *schedulingv1alpha3.PodGroup) (*schedulingv1alpha3.PodGroup, bool) {
	podGroup, ok := wf.podGroups[podGroupKey(pgLookup)]
	return podGroup, ok
}

// getCompositePodGroup returns the current CompositePodGroup object for a given lookup.
func (wf *workloadForest) getCompositePodGroup(cpgLookup *schedulingv1alpha3.CompositePodGroup) (*schedulingv1alpha3.CompositePodGroup, bool) {
	podGroup, ok := wf.compositePodGroups[compositePodGroupKey(cpgLookup)]
	return podGroup, ok
}

func (wf *workloadForest) buildQueuedPodGroupInfo(root any) *framework.QueuedPodGroupInfo {
	switch r := root.(type) {
	case *schedulingv1alpha3.PodGroup:
		return &framework.QueuedPodGroupInfo{
			PodGroupInfo: &framework.PodGroupInfo{
				Namespace:       r.Namespace,
				Name:            r.Name,
				Type:            framework.PodGroupKeyType,
				PodGroup:        r,
				UnscheduledPods: []*v1.Pod{},
			},
			QueuedPodInfos: []*framework.QueuedPodInfo{},
			// Nie brakuje timestampow????

		}
	case *schedulingv1alpha3.CompositePodGroup:
		pgqi := &framework.QueuedPodGroupInfo{
			PodGroupInfo: &framework.PodGroupInfo{
				Namespace:         r.Namespace,
				Name:              r.Name,
				Type:              framework.CompositePodGroupKeyType,
				CompositePodGroup: r,
				Children:          make([]fwk.PodGroupInfo, 0),
				UnscheduledPods:   []*v1.Pod{},
			},
			QueuedPodInfos: []*framework.QueuedPodInfo{},
			// Nie brakuje timestampow????
		}

		key := compositePodGroupKey(r)
		childrenSet, ok := wf.children[key]
		if !ok {
			return pgqi
		}
		for childKey := range childrenSet {
			if childPG, ok := wf.podGroups[childKey]; ok {
				pgqi.PodGroupInfo.Children = append(pgqi.PodGroupInfo.Children, wf.buildQueuedPodGroupInfo(childPG))
			}
			if childCPG, ok := wf.compositePodGroups[childKey]; ok {
				pgqi.PodGroupInfo.Children = append(pgqi.PodGroupInfo.Children, wf.buildQueuedPodGroupInfo(childCPG))
			}
		}
		return pgqi
	default:
		return nil
	}
}
