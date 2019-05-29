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

package admission

import (
	"k8s.io/apimachinery/pkg/api/meta"
)

// fieldManagerAdmissionWrapper is an instance of admission.NamedHandler that wraps other admission handlers for keeping track of their changes and updating their managedFields accordingly
type fieldManagerAdmissionWrapper struct {
	Interface

	manager string
}

// NewFieldManagerAdmissionWrapper creates a new handler wrapping the passed in handler
func NewFieldManagerAdmissionWrapper(handler Interface, manager string) Interface {
	return &fieldManagerAdmissionWrapper{
		Interface: handler,
		manager:   manager,
	}
}

// Admit makes a copy of the current object before running the wrapped admission control check and applies the changes made by the check through the fieldManagers Update function to ensure a proper managedFields state
func (h *fieldManagerAdmissionWrapper) Admit(a Attributes, o ObjectInterfaces) error {
	mutator, ok := h.Interface.(MutationInterface)
	if !ok {
		return nil
	}

	obj := a.GetObject()
	// passthrough if there is no fieldmanager/the feature is not enabled
	if a.GetFieldManager() == nil || obj == nil {
		return mutator.Admit(a, o)
	}

	currentObj := obj.DeepCopyObject()
	if err := mutator.Admit(a, o); err != nil {
		return err
	}

	updatedObj, err := a.GetFieldManager().Update(currentObj, obj, h.manager)
	if err != nil {
		return err
	}

	currentMeta, err := meta.Accessor(obj)
	if err != nil {
		return err
	}
	updatedMeta, err := meta.Accessor(updatedObj)
	if err != nil {
		return err
	}

	currentMeta.SetManagedFields(updatedMeta.GetManagedFields())

	return nil
}
