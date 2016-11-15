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

package poddisruptionbudget

import (
	"fmt"
	"reflect"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/policy"
	"k8s.io/kubernetes/pkg/apis/policy/validation"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
	"k8s.io/kubernetes/pkg/util/validation/field"
)

// podDisruptionBudgetStrategy implements verification logic for PodDisruptionBudgets.
type podDisruptionBudgetStrategy struct {
	runtime.ObjectTyper
	api.NameGenerator
}

// Strategy is the default logic that applies when creating and updating PodDisruptionBudget objects.
var Strategy = podDisruptionBudgetStrategy{api.Scheme, api.SimpleNameGenerator}

// NamespaceScoped returns true because all PodDisruptionBudget' need to be within a namespace.
func (podDisruptionBudgetStrategy) NamespaceScoped() bool {
	return true
}

// PrepareForCreate clears the status of an PodDisruptionBudget before creation.
func (podDisruptionBudgetStrategy) PrepareForCreate(ctx api.Context, obj runtime.Object) {
	podDisruptionBudget := obj.(*policy.PodDisruptionBudget)
	// create cannot set status
	podDisruptionBudget.Status = policy.PodDisruptionBudgetStatus{}

	podDisruptionBudget.Generation = 1
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (podDisruptionBudgetStrategy) PrepareForUpdate(ctx api.Context, obj, old runtime.Object) {
	newPodDisruptionBudget := obj.(*policy.PodDisruptionBudget)
	oldPodDisruptionBudget := old.(*policy.PodDisruptionBudget)
	// Update is not allowed to set status
	newPodDisruptionBudget.Status = oldPodDisruptionBudget.Status

	// Any changes to the spec increment the generation number, any changes to the
	// status should reflect the generation number of the corresponding object.
	// See api.ObjectMeta description for more information on Generation.
	if !reflect.DeepEqual(oldPodDisruptionBudget.Spec, newPodDisruptionBudget.Spec) {
		newPodDisruptionBudget.Generation = oldPodDisruptionBudget.Generation + 1
	}
}

// Validate validates a new PodDisruptionBudget.
func (podDisruptionBudgetStrategy) Validate(ctx api.Context, obj runtime.Object) field.ErrorList {
	podDisruptionBudget := obj.(*policy.PodDisruptionBudget)
	return validation.ValidatePodDisruptionBudget(podDisruptionBudget)
}

// Canonicalize normalizes the object after validation.
func (podDisruptionBudgetStrategy) Canonicalize(obj runtime.Object) {
}

// AllowCreateOnUpdate is true for PodDisruptionBudget; this means you may create one with a PUT request.
func (podDisruptionBudgetStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (podDisruptionBudgetStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) field.ErrorList {
	validationErrorList := validation.ValidatePodDisruptionBudget(obj.(*policy.PodDisruptionBudget))
	updateErrorList := validation.ValidatePodDisruptionBudgetUpdate(obj.(*policy.PodDisruptionBudget), old.(*policy.PodDisruptionBudget))
	return append(validationErrorList, updateErrorList...)
}

// AllowUnconditionalUpdate is the default update policy for PodDisruptionBudget objects. Status update should
// only be allowed if version match.
func (podDisruptionBudgetStrategy) AllowUnconditionalUpdate() bool {
	return false
}

// PodDisruptionBudgetToSelectableFields returns a field set that represents the object.
func PodDisruptionBudgetToSelectableFields(podDisruptionBudget *policy.PodDisruptionBudget) fields.Set {
	return generic.ObjectMetaFieldsSet(&podDisruptionBudget.ObjectMeta, true)
}

// MatchPodDisruptionBudget is the filter used by the generic etcd backend to watch events
// from etcd to clients of the apiserver only interested in specific labels/fields.
func MatchPodDisruptionBudget(label labels.Selector, field fields.Selector) storage.SelectionPredicate {
	return storage.SelectionPredicate{
		Label: label,
		Field: field,
		GetAttrs: func(obj runtime.Object) (labels.Set, fields.Set, error) {
			podDisruptionBudget, ok := obj.(*policy.PodDisruptionBudget)
			if !ok {
				return nil, nil, fmt.Errorf("given object is not a PodDisruptionBudget.")
			}
			return labels.Set(podDisruptionBudget.ObjectMeta.Labels), PodDisruptionBudgetToSelectableFields(podDisruptionBudget), nil
		},
	}
}

type podDisruptionBudgetStatusStrategy struct {
	podDisruptionBudgetStrategy
}

var StatusStrategy = podDisruptionBudgetStatusStrategy{Strategy}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update of status
func (podDisruptionBudgetStatusStrategy) PrepareForUpdate(ctx api.Context, obj, old runtime.Object) {
	newPodDisruptionBudget := obj.(*policy.PodDisruptionBudget)
	oldPodDisruptionBudget := old.(*policy.PodDisruptionBudget)
	// status changes are not allowed to update spec
	newPodDisruptionBudget.Spec = oldPodDisruptionBudget.Spec
}

// ValidateUpdate is the default update validation for an end user updating status
func (podDisruptionBudgetStatusStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) field.ErrorList {
	// TODO: Validate status updates.
	return field.ErrorList{}
	// return validation.ValidatePodDisruptionBudgetStatusUpdate(obj.(*policy.PodDisruptionBudget), old.(*policy.PodDisruptionBudget))
}
