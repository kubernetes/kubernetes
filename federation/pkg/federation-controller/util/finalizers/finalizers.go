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
	"k8s.io/apimachinery/pkg/util/sets"
)

// HasFinalizer returns true if the given object has the given finalizer in its ObjectMeta.
func HasFinalizer(obj runtime.Object, finalizer string) (bool, error) {
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return false, err
	}
	finalizers := sets.NewString(accessor.GetFinalizers()...)
	return finalizers.Has(finalizer), nil
}

// AddFinalizers adds the given finalizers to the given objects ObjectMeta.
// Returns true if the object was updated.
func AddFinalizers(obj runtime.Object, newFinalizers sets.String) (bool, error) {
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return false, err
	}
	oldFinalizers := sets.NewString(accessor.GetFinalizers()...)
	if oldFinalizers.IsSuperset(newFinalizers) {
		return false, nil
	}
	allFinalizers := oldFinalizers.Union(newFinalizers)
	accessor.SetFinalizers(allFinalizers.List())
	return true, nil
}

// RemoveFinalizers removes the given finalizers from the given objects ObjectMeta.
// Returns true if the object was updated.
func RemoveFinalizers(obj runtime.Object, finalizers sets.String) (bool, error) {
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return false, err
	}
	oldFinalizers := sets.NewString(accessor.GetFinalizers()...)
	if oldFinalizers.Intersection(finalizers).Len() == 0 {
		return false, nil
	}
	newFinalizers := oldFinalizers.Difference(finalizers)
	accessor.SetFinalizers(newFinalizers.List())
	return true, nil
}
