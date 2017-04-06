/*
Copyright 2017 The Kubernetes Authors.

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

// Helper functions for manipulating finalizers.
package finalizers

import (
	meta "k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	sliceutils "k8s.io/kubernetes/federation/pkg/federation-controller/util/slice"
)

// Returns true if the given object has the given finalizer in its ObjectMeta.
func HasFinalizer(obj runtime.Object, finalizer string) (bool, error) {
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return false, err
	}
	finalizers := accessor.GetFinalizers()
	for i := range finalizers {
		if finalizers[i] == finalizer {
			return true, nil
		}
	}
	return false, nil
}

// Adds the given finalizers to the given objects ObjectMeta.
// Returns true if the object was updated.
func AddFinalizers(obj runtime.Object, finalizers []string) (bool, error) {
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return false, err
	}
	oldFinalizers := accessor.GetFinalizers()
	newFinalizers := oldFinalizers
	isUpdated := false
	for i := range finalizers {
		if !sliceutils.ContainsString(oldFinalizers, finalizers[i]) {
			newFinalizers = append(newFinalizers, finalizers[i])
			isUpdated = true
		}
	}
	if isUpdated {
		accessor.SetFinalizers(append(accessor.GetFinalizers(), newFinalizers...))
	}
	return isUpdated, nil
}

// Removes the given finalizers from the given objects ObjectMeta.
// Returns true if the object was updated.
func RemoveFinalizers(obj runtime.Object, finalizers []string) (bool, error) {
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return false, err
	}
	oldFinalizers := accessor.GetFinalizers()
	newFinalizers := []string{}
	isUpdated := false
	for i := range oldFinalizers {
		if !sliceutils.ContainsString(finalizers, oldFinalizers[i]) {
			newFinalizers = append(newFinalizers, oldFinalizers[i])
		} else {
			isUpdated = true
		}
	}
	if isUpdated {
		accessor.SetFinalizers(append(accessor.GetFinalizers(), newFinalizers...))
	}
	return isUpdated, nil
}
