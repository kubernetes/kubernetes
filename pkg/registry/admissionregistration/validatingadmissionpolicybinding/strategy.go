/*
Copyright 2022 The Kubernetes Authors.

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

package validatingadmissionpolicybinding

import (
	"context"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/admissionregistration"
	"k8s.io/kubernetes/pkg/apis/admissionregistration/validation"
)

// ValidatingAdmissionPolicyBindingStrategy implements verification logic for ValidatingAdmissionPolicyBinding.
type ValidatingAdmissionPolicyBindingStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating ValidatingAdmissionPolicyBinding objects.
var Strategy = ValidatingAdmissionPolicyBindingStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

// NamespaceScoped returns false because ValidatingAdmissionPolicyBinding is cluster-scoped resource.
func (ValidatingAdmissionPolicyBindingStrategy) NamespaceScoped() bool {
	return false
}

// PrepareForCreate clears the status of an ValidatingAdmissionPolicyBinding before creation.
func (ValidatingAdmissionPolicyBindingStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	ic := obj.(*admissionregistration.ValidatingAdmissionPolicyBinding)
	ic.Generation = 1
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (ValidatingAdmissionPolicyBindingStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newIC := obj.(*admissionregistration.ValidatingAdmissionPolicyBinding)
	oldIC := old.(*admissionregistration.ValidatingAdmissionPolicyBinding)

	// Any changes to the spec increment the generation number, any changes to the
	// status should reflect the generation number of the corresponding object.
	// See metav1.ObjectMeta description for more information on Generation.
	if !apiequality.Semantic.DeepEqual(oldIC.Spec, newIC.Spec) {
		newIC.Generation = oldIC.Generation + 1
	}
}

// Validate validates a new ValidatingAdmissionPolicyBinding.
func (ValidatingAdmissionPolicyBindingStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	return validation.ValidateValidatingAdmissionPolicyBinding(obj.(*admissionregistration.ValidatingAdmissionPolicyBinding))
}

// WarningsOnCreate returns warnings for the creation of the given object.
func (ValidatingAdmissionPolicyBindingStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

// Canonicalize normalizes the object after validation.
func (ValidatingAdmissionPolicyBindingStrategy) Canonicalize(obj runtime.Object) {
}

// AllowCreateOnUpdate is true for ValidatingAdmissionPolicyBinding; this means you may create one with a PUT request.
func (ValidatingAdmissionPolicyBindingStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (ValidatingAdmissionPolicyBindingStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateValidatingAdmissionPolicyBindingUpdate(obj.(*admissionregistration.ValidatingAdmissionPolicyBinding), old.(*admissionregistration.ValidatingAdmissionPolicyBinding))
}

// WarningsOnUpdate returns warnings for the given update.
func (ValidatingAdmissionPolicyBindingStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

// AllowUnconditionalUpdate is the default update policy for ValidatingAdmissionPolicyBinding objects. Status update should
// only be allowed if version match.
func (ValidatingAdmissionPolicyBindingStrategy) AllowUnconditionalUpdate() bool {
	return false
}
