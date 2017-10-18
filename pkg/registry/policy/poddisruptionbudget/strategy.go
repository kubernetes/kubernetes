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
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/policy"
	"k8s.io/kubernetes/pkg/apis/policy/validation"
)

// podDisruptionBudgetStrategy implements verification logic for PodDisruptionBudgets.
type podDisruptionBudgetStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating PodDisruptionBudget objects.
var Strategy = podDisruptionBudgetStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

// NamespaceScoped returns true because all PodDisruptionBudget' need to be within a namespace.
func (podDisruptionBudgetStrategy) NamespaceScoped() bool {
	return true
}

// PrepareForCreate clears the status of an PodDisruptionBudget before creation.
func (podDisruptionBudgetStrategy) PrepareForCreate(ctx genericapirequest.Context, obj runtime.Object) {
	podDisruptionBudget := obj.(*policy.PodDisruptionBudget)
	// create cannot set status
	podDisruptionBudget.Status = policy.PodDisruptionBudgetStatus{}

	podDisruptionBudget.Generation = 1
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (podDisruptionBudgetStrategy) PrepareForUpdate(ctx genericapirequest.Context, obj, old runtime.Object) {
	newPodDisruptionBudget := obj.(*policy.PodDisruptionBudget)
	oldPodDisruptionBudget := old.(*policy.PodDisruptionBudget)
	// Update is not allowed to set status
	newPodDisruptionBudget.Status = oldPodDisruptionBudget.Status

	// Any changes to the spec increment the generation number, any changes to the
	// status should reflect the generation number of the corresponding object.
	// See metav1.ObjectMeta description for more information on Generation.
	if !apiequality.Semantic.DeepEqual(oldPodDisruptionBudget.Spec, newPodDisruptionBudget.Spec) {
		newPodDisruptionBudget.Generation = oldPodDisruptionBudget.Generation + 1
	}
}

// Validate validates a new PodDisruptionBudget.
func (podDisruptionBudgetStrategy) Validate(ctx genericapirequest.Context, obj runtime.Object) field.ErrorList {
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
func (podDisruptionBudgetStrategy) ValidateUpdate(ctx genericapirequest.Context, obj, old runtime.Object) field.ErrorList {
	validationErrorList := validation.ValidatePodDisruptionBudget(obj.(*policy.PodDisruptionBudget))
	updateErrorList := validation.ValidatePodDisruptionBudgetUpdate(obj.(*policy.PodDisruptionBudget), old.(*policy.PodDisruptionBudget))
	return append(validationErrorList, updateErrorList...)
}

// AllowUnconditionalUpdate is the default update policy for PodDisruptionBudget objects. Status update should
// only be allowed if version match.
func (podDisruptionBudgetStrategy) AllowUnconditionalUpdate() bool {
	return false
}

type podDisruptionBudgetStatusStrategy struct {
	podDisruptionBudgetStrategy
}

var StatusStrategy = podDisruptionBudgetStatusStrategy{Strategy}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update of status
func (podDisruptionBudgetStatusStrategy) PrepareForUpdate(ctx genericapirequest.Context, obj, old runtime.Object) {
	newPodDisruptionBudget := obj.(*policy.PodDisruptionBudget)
	oldPodDisruptionBudget := old.(*policy.PodDisruptionBudget)
	// status changes are not allowed to update spec
	newPodDisruptionBudget.Spec = oldPodDisruptionBudget.Spec
}

// ValidateUpdate is the default update validation for an end user updating status
func (podDisruptionBudgetStatusStrategy) ValidateUpdate(ctx genericapirequest.Context, obj, old runtime.Object) field.ErrorList {
	// TODO: Validate status updates.
	return field.ErrorList{}
	// return validation.ValidatePodDisruptionBudgetStatusUpdate(obj.(*policy.PodDisruptionBudget), old.(*policy.PodDisruptionBudget))
}
