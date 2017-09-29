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

package util

import (
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/util/initialization"
	"k8s.io/apiserver/pkg/admission"
)

// IsUpdatingInitializedObject returns true if the operation is trying to update
// an already initialized object.
func IsUpdatingInitializedObject(a admission.Attributes) (bool, error) {
	if a.GetOperation() != admission.Update {
		return false, nil
	}
	oldObj := a.GetOldObject()
	accessor, err := meta.Accessor(oldObj)
	if err != nil {
		return false, err
	}
	if initialization.IsInitialized(accessor.GetInitializers()) {
		return true, nil
	}
	return false, nil
}

// IsUpdatingUninitializedObject returns true if the operation is trying to
// update an object that is not initialized yet.
func IsUpdatingUninitializedObject(a admission.Attributes) (bool, error) {
	if a.GetOperation() != admission.Update {
		return false, nil
	}
	oldObj := a.GetOldObject()
	accessor, err := meta.Accessor(oldObj)
	if err != nil {
		return false, err
	}
	if initialization.IsInitialized(accessor.GetInitializers()) {
		return false, nil
	}
	return true, nil
}

// IsInitializationCompletion returns true if the operation removes all pending
// initializers.
func IsInitializationCompletion(a admission.Attributes) (bool, error) {
	if a.GetOperation() != admission.Update {
		return false, nil
	}
	oldObj := a.GetOldObject()
	oldInitialized, err := initialization.IsObjectInitialized(oldObj)
	if err != nil {
		return false, err
	}
	if oldInitialized {
		return false, nil
	}
	newObj := a.GetObject()
	newInitialized, err := initialization.IsObjectInitialized(newObj)
	if err != nil {
		return false, err
	}
	return newInitialized, nil
}
