/*
Copyright 2014 The Kubernetes Authors.

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

package petset

import (
	"fmt"
	"reflect"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/apps/validation"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/validation/field"
)

// petSetStrategy implements verification logic for Replication PetSets.
type petSetStrategy struct {
	runtime.ObjectTyper
	api.NameGenerator
}

// Strategy is the default logic that applies when creating and updating Replication PetSet objects.
var Strategy = petSetStrategy{api.Scheme, api.SimpleNameGenerator}

// NamespaceScoped returns true because all PetSet' need to be within a namespace.
func (petSetStrategy) NamespaceScoped() bool {
	return true
}

// PrepareForCreate clears the status of an PetSet before creation.
func (petSetStrategy) PrepareForCreate(ctx api.Context, obj runtime.Object) {
	petSet := obj.(*apps.PetSet)
	// create cannot set status
	petSet.Status = apps.PetSetStatus{}

	petSet.Generation = 1
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (petSetStrategy) PrepareForUpdate(ctx api.Context, obj, old runtime.Object) {
	newPetSet := obj.(*apps.PetSet)
	oldPetSet := old.(*apps.PetSet)
	// Update is not allowed to set status
	newPetSet.Status = oldPetSet.Status

	// Any changes to the spec increment the generation number, any changes to the
	// status should reflect the generation number of the corresponding object.
	// See api.ObjectMeta description for more information on Generation.
	if !reflect.DeepEqual(oldPetSet.Spec, newPetSet.Spec) {
		newPetSet.Generation = oldPetSet.Generation + 1
	}

}

// Validate validates a new PetSet.
func (petSetStrategy) Validate(ctx api.Context, obj runtime.Object) field.ErrorList {
	petSet := obj.(*apps.PetSet)
	return validation.ValidatePetSet(petSet)
}

// Canonicalize normalizes the object after validation.
func (petSetStrategy) Canonicalize(obj runtime.Object) {
}

// AllowCreateOnUpdate is false for PetSet; this means POST is needed to create one.
func (petSetStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (petSetStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) field.ErrorList {
	validationErrorList := validation.ValidatePetSet(obj.(*apps.PetSet))
	updateErrorList := validation.ValidatePetSetUpdate(obj.(*apps.PetSet), old.(*apps.PetSet))
	return append(validationErrorList, updateErrorList...)
}

// AllowUnconditionalUpdate is the default update policy for PetSet objects.
func (petSetStrategy) AllowUnconditionalUpdate() bool {
	return true
}

// PetSetToSelectableFields returns a field set that represents the object.
func PetSetToSelectableFields(petSet *apps.PetSet) fields.Set {
	return generic.ObjectMetaFieldsSet(petSet.ObjectMeta, true)
}

// MatchPetSet is the filter used by the generic etcd backend to watch events
// from etcd to clients of the apiserver only interested in specific labels/fields.
func MatchPetSet(label labels.Selector, field fields.Selector) generic.Matcher {
	return &generic.SelectionPredicate{
		Label: label,
		Field: field,
		GetAttrs: func(obj runtime.Object) (labels.Set, fields.Set, error) {
			petSet, ok := obj.(*apps.PetSet)
			if !ok {
				return nil, nil, fmt.Errorf("given object is not an PetSet.")
			}
			return labels.Set(petSet.ObjectMeta.Labels), PetSetToSelectableFields(petSet), nil
		},
	}
}

type petSetStatusStrategy struct {
	petSetStrategy
}

var StatusStrategy = petSetStatusStrategy{Strategy}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update of status
func (petSetStatusStrategy) PrepareForUpdate(ctx api.Context, obj, old runtime.Object) {
	newPetSet := obj.(*apps.PetSet)
	oldPetSet := old.(*apps.PetSet)
	// status changes are not allowed to update spec
	newPetSet.Spec = oldPetSet.Spec
}

// ValidateUpdate is the default update validation for an end user updating status
func (petSetStatusStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) field.ErrorList {
	// TODO: Validate status updates.
	return validation.ValidatePetSetStatusUpdate(obj.(*apps.PetSet), old.(*apps.PetSet))
}
