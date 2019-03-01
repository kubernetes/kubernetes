/*
Copyright 2019 The Kubernetes Authors.

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

// This file contains structures that implement equivalence class types.
// EquivClass is a thread safe map of unschedulable equivalence hashes.
// Once a pod is marked an unschedulable, its equivalence hash is added
// to the map. Equivalence class improves scheduler velocity and responsiveness
// by avoiding checking the schedulability of all of these pods when one is
// determined unschedulable. This is particularly useful in batch processing.

package equivalence

import (
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"

	"sync"
)

// EquivClass is a thread safe map saves the Previous scheduling information,
// It uses equivhash as the key to determine whether it is not schedulable.
type EquivClass struct {
	equivClassMap *sync.Map
}

// New creates an empty EquivClass.
func New() *EquivClass {
	return &EquivClass{
		equivClassMap: new(sync.Map),
	}
}

// Add adds the equivHash to the equivClassMap.
func (e *EquivClass) Add(equivHash types.UID) {
	e.equivClassMap.Store(equivHash, true)
}

// Get returns whether the equivHash is in the equivClassMap.
func (e *EquivClass) Get(equivHash types.UID) bool {
	_, ok := e.equivClassMap.Load(equivHash)
	return ok
}

// Delete deletes the equivHash from the equivClassMap.
func (e *EquivClass) Delete(equivHash types.UID) {
	e.equivClassMap.Delete(equivHash)
}

// Clear deletes all equivHash from the equivClassMap.
func (e *EquivClass) Clear() {
	e.equivClassMap = new(sync.Map)
}

// GetEquivHash returns the pod's UID of controllerRef.
// NOTE (resouer): To avoid hash collision issue, in alpha stage we decide to use `controllerRef` only to determine
// whether two pods are equivalent. This may change with the feature evolves.
func GetEquivHash(pod *v1.Pod) types.UID {
	ownerReferences := pod.GetOwnerReferences()
	if ownerReferences != nil {
		return ownerReferences[0].UID
	}

	// If pod's UID of controllerRef is nil, return nil.
	return ""
}
