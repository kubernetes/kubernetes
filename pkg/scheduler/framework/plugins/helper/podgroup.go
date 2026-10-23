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

package helper

import (
	"fmt"
	"iter"

	v1 "k8s.io/api/core/v1"
	fwk "k8s.io/kube-scheduler/framework"
)

// MatchingSchedulingGroup returns true if two pods belong to the same scheduling group.
func MatchingSchedulingGroup(pod1, pod2 *v1.Pod) bool {
	return pod1.Namespace == pod2.Namespace &&
		pod1.Spec.SchedulingGroup != nil &&
		pod2.Spec.SchedulingGroup != nil &&
		*pod1.Spec.SchedulingGroup.PodGroupName == *pod2.Spec.SchedulingGroup.PodGroupName
}

// GetPodGroupStates returns an iterator sequence over all leaf PodGroupStates for the given rootEntityKey.
// If rootEntityKey is a CompositePodGroupKeyType, it recursively traverses its children.
// If rootEntityKey is a PodGroupKeyType, it yields its PodGroupState.
func GetPodGroupStates(sharedLister fwk.SharedLister, rootEntityKey fwk.EntityKey) iter.Seq2[fwk.PodGroupState, error] {
	return func(yield func(fwk.PodGroupState, error) bool) {
		var walk func(entityKey fwk.EntityKey) bool
		walk = func(entityKey fwk.EntityKey) bool {
			if entityKey.GetType() == fwk.CompositePodGroupKeyType {
				cpgState, err := sharedLister.CompositePodGroupStates().Get(entityKey.GetNamespace(), entityKey.GetName())
				if err != nil {
					return yield(nil, err)
				}
				for _, childEntityKey := range cpgState.GetChildren() {
					if !walk(childEntityKey) {
						return false
					}
				}
				return true
			}
			pgState, err := sharedLister.PodGroupStates().Get(entityKey.GetNamespace(), entityKey.GetName())
			if err != nil {
				return yield(nil, err)
			}
			return yield(pgState, nil)
		}
		walk(rootEntityKey)
	}
}

// TraverseHierarchyUp returns an iterator over the groups on the path from startKey to the root of its
// PodGroup/CompositePodGroup hierarchy, starting with startKey itself.
// The traversal ends by yielding an error when a group cannot be resolved or the key type is unsupported.
// Feature gating is left to the listers: when GenericWorkload or CompositePodGroup is disabled,
// the corresponding lister rejects lookups, so the traversal ends there just like for a missing group.
func TraverseHierarchyUp(snapshot fwk.SharedLister, startKey fwk.EntityKey) iter.Seq2[*fwk.GenericPodGroup, error] {
	return func(yield func(*fwk.GenericPodGroup, error) bool) {
		currentKey := startKey
		for {
			var gpg *fwk.GenericPodGroup
			switch currentKey.GetType() {
			case fwk.PodGroupKeyType:
				pg, err := snapshot.PodGroups().Get(currentKey.GetNamespace(), currentKey.GetName())
				if err != nil {
					yield(nil, err)
					return
				}
				gpg = fwk.NewGenericPodGroup(pg)
			case fwk.CompositePodGroupKeyType:
				cpg, err := snapshot.CompositePodGroups().Get(currentKey.GetNamespace(), currentKey.GetName())
				if err != nil {
					yield(nil, err)
					return
				}
				gpg = fwk.NewGenericCompositePodGroup(cpg)
			default:
				yield(nil, fmt.Errorf("unsupported key type %q for %v", currentKey.GetType(), currentKey))
				return
			}

			if !yield(gpg, nil) {
				return
			}

			parentKey, hasParent := gpg.GetParentKey()
			if !hasParent {
				return
			}
			currentKey = parentKey
		}
	}
}
