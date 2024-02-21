/*
Copyright 2024 The Kubernetes Authors.

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

package typetracker

import (
	"sync"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
)

// Instance keeps a bidirectional mapping of policy <-> referred GVKs.
type Instance struct {
	// map of policyName -> referred GVKs
	policyToGVKs map[string]sets.Set[schema.GroupVersionKind]
	// map of referred GVK to policies
	gvkToPolicies map[schema.GroupVersionKind]sets.Set[string]

	mu sync.Mutex
}

// New creates an empty type tracker.
func New() *Instance {
	return &Instance{
		policyToGVKs:  make(map[string]sets.Set[schema.GroupVersionKind]),
		gvkToPolicies: make(map[schema.GroupVersionKind]sets.Set[string]),
	}
}

// ObserveChange observes the creation/update of a policy. The tracker updates its internal bidirectional
// policy from/to GVKs mapping.
func (t *Instance) ObserveChange(policyName string, referredGVKs []schema.GroupVersionKind) {
	t.mu.Lock()
	defer t.mu.Unlock()

	oldGVKsSet, ok := t.policyToGVKs[policyName]
	if !ok {
		oldGVKsSet = sets.New[schema.GroupVersionKind]()
	}
	newGVKsSet := sets.New[schema.GroupVersionKind](referredGVKs...)
	toRemove := oldGVKsSet.Difference(newGVKsSet)
	toAdd := newGVKsSet.Difference(oldGVKsSet)
	for gvk := range toRemove {
		if s, ok := t.gvkToPolicies[gvk]; ok {
			s.Delete(policyName)
			if s.Len() == 0 {
				delete(t.gvkToPolicies, gvk)
			}
		}
	}
	for gvk := range toAdd {
		s, ok := t.gvkToPolicies[gvk]
		if !ok {
			s = sets.New[string](policyName)
			t.gvkToPolicies[gvk] = s
		}
		s.Insert(policyName)
	}
	t.policyToGVKs[policyName] = newGVKsSet
}

// ObservedDeletion observes the deletion of a policy. The tracker updates its internal bidirectional
// policy from/to GVKs mapping.
func (t *Instance) ObservedDeletion(policyName string) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if oldGVKsSet, ok := t.policyToGVKs[policyName]; ok {
		for gvk := range oldGVKsSet {
			if s, ok := t.gvkToPolicies[gvk]; ok {
				s.Delete(policyName)
				if s.Len() == 0 {
					delete(t.gvkToPolicies, gvk)
				}
			}
		}
	}
	delete(t.policyToGVKs, policyName)
}

// AffectedPolicies find all policies that are affected due to the change of the give list of GVKs.
func (t *Instance) AffectedPolicies(changedGVKs []schema.GroupVersionKind) []string {
	t.mu.Lock()
	defer t.mu.Unlock()

	result := sets.New[string]()
	for _, gvk := range changedGVKs {
		if policies, ok := t.gvkToPolicies[gvk]; ok {
			// insert one-by-one instead of Set[T].Union to avoid the copy from the latter,
			// which otherwise increase the whole operation to O(N^2) from O(N)
			for policy := range policies {
				result.Insert(policy)
			}
		}
	}
	return result.UnsortedList()
}
