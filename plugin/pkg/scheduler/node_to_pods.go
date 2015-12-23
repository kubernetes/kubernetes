/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package scheduler

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/sets"
)

type PodSet map[*api.Pod]sets.Empty

// Insert inserts all items into the set.
func (s PodSet) Insert(items ...*api.Pod) {
	for _, item := range items {
		s[item] = sets.Empty{}
	}
}

// Delete removes all items from the set.
func (s PodSet) Delete(items ...*api.Pod) {
	for _, item := range items {
		delete(s, item)
	}
}

// Has returns true if and only if item is contained in the set.
func (s PodSet) Has(item *api.Pod) bool {
	_, contained := s[item]
	return contained
}

type nodeToPods map[string]PodSet

func (m nodeToPods) Insert(pod *api.Pod) {
	m[pod.Spec.NodeName].Insert(pod)
}

func (m nodeToPods) Delete(pod *api.Pod) {
	m[pod.Spec.NodeName].Delete(pod)
}

type LookupTable struct {
	nodeToPods nodeToPods
	actionLocker
}

func (t *LookupTable) AddPod(pod *api.Pod) {
	t.nodeToPods[pod.Spec.NodeName].Insert(pod)
}

func (t *LookupTable) RemovePod(pod *api.Pod) {
	t.nodeToPods[pod.Spec.NodeName].Delete(pod)
}

func (t *LookupTable) Update(oldPod, newPod *api.Pod) {
	if oldPod.Spec.NodeName != newPod.Spec.NodeName {
		t.nodeToPods.Delete(oldPod)
		t.nodeToPods.Insert(newPod)
	}
}

func (t *LookupTable) GetPodsOnNode(node string) []*api.Pod {
	var pods []*api.Pod
	set := t.nodeToPods[node]
	for pod := range set {
		pods = append(pods, pod)
	}
	return pods
}
