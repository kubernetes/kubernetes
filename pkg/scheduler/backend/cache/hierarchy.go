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

package cache

import (
	"errors"
	"fmt"

	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	fwk "k8s.io/kube-scheduler/framework"
)

// podGroupStateItem represents a PodGroup state entry that exposes its API object and runtime state.
type podGroupStateItem interface {
	fwk.PodGroupState
	PodGroup() *schedulingv1beta1.PodGroup
}

// compositePodGroupStateItem represents a CompositePodGroup state entry that exposes its API object and runtime state.
type compositePodGroupStateItem interface {
	fwk.CompositePodGroupState
	CompositePodGroup() *schedulingv1alpha3.CompositePodGroup
}

var _ podGroupStateItem = &podGroupState{}
var _ podGroupStateItem = &podGroupStateSnapshot{}
var _ compositePodGroupStateItem = &compositePodGroupState{}
var _ compositePodGroupStateItem = &compositePodGroupStateSnapshot{}

// hierarchyWrapper provides unified hierarchy traversal over PodGroup and CompositePodGroup state stores.
// It is generic over the state item types so that the live cache states and the immutable snapshot
// states share a single traversal implementation.
//
// The wrapper borrows the caller's maps rather than copying them, so the caller must hold the lock
// protecting those maps for the entire lifetime of the wrapper.
type hierarchyWrapper[PG podGroupStateItem, CPG compositePodGroupStateItem] struct {
	podGroupStates           map[fwk.EntityKey]PG
	compositePodGroupStates  map[fwk.EntityKey]CPG
	compositePodGroupEnabled bool
}

// newHierarchyWrapper creates a new hierarchyWrapper.
func newHierarchyWrapper[PG podGroupStateItem, CPG compositePodGroupStateItem](
	pgStates map[fwk.EntityKey]PG,
	cpgStates map[fwk.EntityKey]CPG,
	cpgEnabled bool,
) *hierarchyWrapper[PG, CPG] {
	return &hierarchyWrapper[PG, CPG]{
		podGroupStates:           pgStates,
		compositePodGroupStates:  cpgStates,
		compositePodGroupEnabled: cpgEnabled,
	}
}

// errGroupNotFound marks a missing link in the hierarchy.
var errGroupNotFound = errors.New("not found in the hierarchy")

// findRootKey walks up the hierarchy from key and returns its root EntityKey. It never returns a
// nil key without an error.
func (hw *hierarchyWrapper[PG, CPG]) findRootKey(key fwk.EntityKey) (*fwk.EntityKey, error) {
	currentKey := key
	for range schedulingv1alpha3.WorkloadMaxTreeDepth {
		switch currentKey.Type {
		case fwk.PodGroupKeyType:
			pgs, ok := hw.podGroupStates[currentKey]
			if !ok {
				return nil, fmt.Errorf("pod group state for %s: %w", currentKey.String(), errGroupNotFound)
			}
			pg := pgs.PodGroup()
			if pg == nil {
				return nil, fmt.Errorf("pod group object in state for %s: %w", currentKey.String(), errGroupNotFound)
			}
			if !hw.compositePodGroupEnabled || pg.Spec.ParentCompositePodGroupName == nil {
				return &currentKey, nil
			}
			currentKey = fwk.CompositePodGroupKey(pg.Namespace, *pg.Spec.ParentCompositePodGroupName)
		case fwk.CompositePodGroupKeyType:
			cpgs, ok := hw.compositePodGroupStates[currentKey]
			if !ok {
				return nil, fmt.Errorf("composite pod group state for %s: %w", currentKey.String(), errGroupNotFound)
			}
			cpg := cpgs.CompositePodGroup()
			if cpg == nil {
				return nil, fmt.Errorf("composite pod group object in state for %s: %w", currentKey.String(), errGroupNotFound)
			}
			// The parent link is only meaningful while the feature is enabled. With the feature
			// off nothing populates the CompositePodGroup states in the first place, so this is
			// unreachable in practice, but following the link would be wrong if it ever were.
			if !hw.compositePodGroupEnabled || cpg.Spec.ParentCompositePodGroupName == nil {
				return &currentKey, nil
			}
			currentKey = fwk.CompositePodGroupKey(cpg.Namespace, *cpg.Spec.ParentCompositePodGroupName)
		default:
			return nil, fmt.Errorf("unsupported key type %s in hierarchy traversal for %s", currentKey.Type, currentKey.String())
		}
	}
	return nil, fmt.Errorf("hierarchy starting at %s is deeper than the maximum of %d, or contains a cycle", key.String(), schedulingv1alpha3.WorkloadMaxTreeDepth)
}

// FindRootKeyForGroup returns the root *EntityKey of the hierarchy for the given EntityKey,
// or nil if the root group was not found (i.e. does not exist).
// The key must be of PodGroupKey or CompositePodGroupKey type.
func (hw *hierarchyWrapper[PG, CPG]) FindRootKeyForGroup(key fwk.EntityKey) (*fwk.EntityKey, error) {
	rootKey, err := hw.findRootKey(key)
	if errors.Is(err, errGroupNotFound) {
		return nil, nil
	}
	return rootKey, err
}

// FindRootGroup returns the *RootGroup containing the root key, PodGroup/PodGroupState (if root is a PodGroup),
// or CompositePodGroup/CompositePodGroupState (if root is a CompositePodGroup) for the given EntityKey,
// or nil if the root group was not found.
func (hw *hierarchyWrapper[PG, CPG]) FindRootGroup(key fwk.EntityKey) (*fwk.RootGroup, error) {
	rootKey, err := hw.FindRootKeyForGroup(key)
	if err != nil {
		return nil, err
	}
	if rootKey == nil {
		return nil, nil
	}

	switch rootKey.Type {
	case fwk.PodGroupKeyType:
		pgs, ok := hw.podGroupStates[*rootKey]
		if !ok || pgs.PodGroup() == nil {
			return nil, nil
		}
		return &fwk.RootGroup{
			GenericPodGroup: fwk.NewGenericPodGroup(pgs.PodGroup()),
			PodGroupState:   pgs,
		}, nil
	case fwk.CompositePodGroupKeyType:
		cpgs, ok := hw.compositePodGroupStates[*rootKey]
		if !ok || cpgs.CompositePodGroup() == nil {
			return nil, nil
		}
		return &fwk.RootGroup{
			GenericPodGroup:        fwk.NewGenericCompositePodGroup(cpgs.CompositePodGroup()),
			CompositePodGroupState: cpgs,
		}, nil
	default:
		return nil, fmt.Errorf("unsupported root key type %s for %s", rootKey.Type, key.String())
	}
}
