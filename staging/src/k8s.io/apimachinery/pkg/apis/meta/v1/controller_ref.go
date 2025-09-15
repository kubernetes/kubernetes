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
	"k8s.io/utils/ptr"
)

// IsControlledBy checks if the  object has a controllerRef set to the given owner
func IsControlledBy(obj Object, owner Object) bool {
	ref := GetControllerOfNoCopy(obj)
	if ref == nil {
		return false
	}
	return ref.UID == owner.GetUID()
}

// GetControllerOf returns a pointer to a copy of the controllerRef if controllee has a controller
func GetControllerOf(controllee Object) *OwnerReference {
	ref := GetControllerOfNoCopy(controllee)
	if ref == nil {
		return nil
	}
	cp := *ref
	cp.Controller = ptr.To(*ref.Controller)
	if ref.BlockOwnerDeletion != nil {
		cp.BlockOwnerDeletion = ptr.To(*ref.BlockOwnerDeletion)
	}
	return &cp
}

// GetControllerOfNoCopy returns a pointer to the controllerRef if controllee has a controller
func GetControllerOfNoCopy(controllee Object) *OwnerReference {
	refs := controllee.GetOwnerReferences()
	for i := range refs {
		if refs[i].Controller != nil && *refs[i].Controller {
			return &refs[i]
		}
	}
	return nil
}

// NewControllerRef creates an OwnerReference pointing to the given owner.
func NewControllerRef(owner Object, gvk schema.GroupVersionKind) *OwnerReference {
	return &OwnerReference{
		APIVersion:         gvk.GroupVersion().String(),
		Kind:               gvk.Kind,
		Name:               owner.GetName(),
		UID:                owner.GetUID(),
		BlockOwnerDeletion: ptr.To(true),
		Controller:         ptr.To(true),
	}
}
