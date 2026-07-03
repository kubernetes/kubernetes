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
)

// workloadForest maintains a consistent view of observed PodGroup objects.
// It ensures that scheduling queue invariants are preserved, independent of
// asynchronous updates happening in the scheduler cache.
// Outside of the scheduling queue, cache should be used as the source of truth.
// This structure is not thread-safe and should be accessed only under the lock of the PriorityQueue.
type workloadForest struct {
	podGroups map[string]*schedulingv1alpha3.PodGroup
}

func newWorkloadForest() *workloadForest {
	return &workloadForest{
		podGroups: make(map[string]*schedulingv1alpha3.PodGroup),
	}
}

// addPodGroup adds a PodGroup to the forest.
func (wf *workloadForest) addPodGroup(podGroup *schedulingv1alpha3.PodGroup) {
	wf.podGroups[podGroupKey(podGroup)] = podGroup
}

// updatePodGroup updates a PodGroup in the forest.
func (wf *workloadForest) updatePodGroup(podGroup *schedulingv1alpha3.PodGroup) {
	wf.podGroups[podGroupKey(podGroup)] = podGroup
}

// deletePodGroup removes a PodGroup from the forest.
func (wf *workloadForest) deletePodGroup(podGroup *schedulingv1alpha3.PodGroup) {
	delete(wf.podGroups, podGroupKey(podGroup))
}

// getRootForPod returns the current root PodGroup object for a given pod.
func (wf *workloadForest) getRootForPod(pod *v1.Pod) (*schedulingv1alpha3.PodGroup, bool) {
	podGroup, ok := wf.podGroups[podGroupKeyForPod(pod)]
	return podGroup, ok
}

// getPodGroup returns the current PodGroup object for a given lookup.
func (wf *workloadForest) getPodGroup(pgLookup *schedulingv1alpha3.PodGroup) (*schedulingv1alpha3.PodGroup, bool) {
	podGroup, ok := wf.podGroups[podGroupKey(pgLookup)]
	return podGroup, ok
}
