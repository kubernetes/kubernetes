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

package v1

import (
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// IsControlledBy checks if the  object has a controllerRef set to the given owner
func IsControlledBy(obj Object, owner Object) bool {
	ref := GetControllerOf(obj)
	if ref == nil {
		return false
	}
	return ref.UID == owner.GetUID()
}

// GetControllerOf returns a pointer to a copy of the controllerRef if controllee has a controller
func GetControllerOf(controllee Object) *OwnerReference {
	for _, ref := range controllee.GetOwnerReferences() {
		if ref.Controller != nil && *ref.Controller {
			return &ref
		}
	}
	return nil
}

// NewNamespacedControllerRef creates an OwnerReference pointing to the given owner, with the resource in addition to the kind.
// The resource must be in the same group as the kind.  The owner and dependent must both be in a namespace.
func NewNamespacedControllerRef(owner Object, gvr schema.GroupVersionResource, kind string) *OwnerReference {
	blockOwnerDeletion := true
	isController := true
	return &OwnerReference{
		APIVersion:         gvr.GroupVersion().String(),
		Kind:               kind,
		Resource:           gvr.Resource,
		Name:               owner.GetName(),
		Namespace:          stringPtr(owner.GetNamespace()),
		UID:                owner.GetUID(),
		BlockOwnerDeletion: &blockOwnerDeletion,
		Controller:         &isController,
	}
}

// NewClusterControllerRef creates an OwnerReference pointing to the given owner, with the resource in addition to the kind.
// The resource must be in the same group as the kind.  The owner must be cluster scoped.
func NewClusterControllerRef(owner Object, gvr schema.GroupVersionResource, kind string) *OwnerReference {
	blockOwnerDeletion := true
	isController := true
	return &OwnerReference{
		APIVersion:         gvr.GroupVersion().String(),
		Kind:               kind,
		Resource:           gvr.Resource,
		Name:               owner.GetName(),
		Namespace:          stringPtr(""),
		UID:                owner.GetUID(),
		BlockOwnerDeletion: &blockOwnerDeletion,
		Controller:         &isController,
	}
}

func stringPtr(in string) *string {
	return &in
}
