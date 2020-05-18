/*
Copyright 2020 The Kubernetes Authors.

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

package meta

import (
	"reflect"

	"k8s.io/apimachinery/pkg/runtime"
)

// DeepCopiableObject is a runtime.Object with get/set finalizer functions
type FinalizeableObject interface {
	runtime.Object
	Finalizeable
}

// Finalizeable has get/set finalizer functions
type Finalizeable interface {
	GetFinalizers() []string
	SetFinalizers(finalizers []string)
}

// AddFinalizer adds the finalizer to the metadata if it isn't already present.  If it is present, no action is taken.
// If no action is taken, the return value is the passed object.  If the finalizer is added, the object is deep copied
// to avoid mutating input.
func AddFinalizer(object FinalizeableObject, finalizerName string) runtime.Object {
	if HasFinalizer(object, finalizerName) {
		return object
	}

	ret := object.DeepCopyObject().(FinalizeableObject)
	ret.SetFinalizers(append(ret.GetFinalizers(), finalizerName))
	return ret
}

// RemoveFinalizer removes the finalizer to the metadata if it is present.  If it is not present, no action is taken.
// If no action is taken, the return value is the passed object.  If the finalizer is added, the object is deep copied
// to avoid mutating input.
func RemoveFinalizer(object FinalizeableObject, finalizerName string) runtime.Object {
	if !HasFinalizer(object, finalizerName) {
		return object
	}

	ret := object.DeepCopyObject().(FinalizeableObject)
	ret.SetFinalizers(removeString(ret.GetFinalizers(), finalizerName))
	return ret
}

// HasFinalizer returns true if the specified finalizer is in the list of finalizers.
func HasFinalizer(object Finalizeable, finalizerName string) bool {
	for _, finalizer := range object.GetFinalizers() {
		if finalizer == finalizerName {
			return true
		}
	}
	return false
}

// FinalizersEqual is syntactic sugar for determining if finalizers were mutated while allowing type preservation.
// eg.
//   updated := meta.AddFinalizer(service, key).(*v1.Service)
//   if meta.FinalizersEqual(updated, service) {
//       return nil
//   }
func FinalizersEqual(newObj, oldObj FinalizeableObject) bool {
	return reflect.DeepEqual(newObj, oldObj)
}

// removeString returns a newly created []string that contains all items from slice that are not equal to s.
func removeString(slice []string, s string) []string {
	var newSlice []string
	for _, item := range slice {
		if item != s {
			newSlice = append(newSlice, item)
		}
	}
	return newSlice
}
