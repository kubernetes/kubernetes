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

package validatingadmissionpolicy

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

// validatingAdmissionPolicyStrategy implements verification logic for ValidatingAdmissionPolicy.
type validatingAdmissionPolicyStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating validatingAdmissionPolicy objects.
var Strategy = validatingAdmissionPolicyStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

// NamespaceScoped returns false because ValidatingAdmissionPolicy is cluster-scoped resource.
func (validatingAdmissionPolicyStrategy) NamespaceScoped() bool {
	return false
}

// PrepareForCreate clears the status of an validatingAdmissionPolicy before creation.
func (validatingAdmissionPolicyStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	ic := obj.(*admissionregistration.ValidatingAdmissionPolicy)
	ic.Generation = 1
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (validatingAdmissionPolicyStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newIC := obj.(*admissionregistration.ValidatingAdmissionPolicy)
	oldIC := old.(*admissionregistration.ValidatingAdmissionPolicy)

	// Any changes to the spec increment the generation number, any changes to the
	// status should reflect the generation number of the corresponding object.
	// See metav1.ObjectMeta description for more information on Generation.
	if !apiequality.Semantic.DeepEqual(oldIC.Spec, newIC.Spec) {
		newIC.Generation = oldIC.Generation + 1
	}
}

// Validate validates a new validatingAdmissionPolicy.
func (validatingAdmissionPolicyStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	return validation.ValidateValidatingAdmissionPolicy(obj.(*admissionregistration.ValidatingAdmissionPolicy))
}

// WarningsOnCreate returns warnings for the creation of the given object.
func (validatingAdmissionPolicyStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

// Canonicalize normalizes the object after validation.
func (validatingAdmissionPolicyStrategy) Canonicalize(obj runtime.Object) {
}

// AllowCreateOnUpdate is true for validatingAdmissionPolicy; this means you may create one with a PUT request.
func (validatingAdmissionPolicyStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (validatingAdmissionPolicyStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateValidatingAdmissionPolicyUpdate(obj.(*admissionregistration.ValidatingAdmissionPolicy), old.(*admissionregistration.ValidatingAdmissionPolicy))
}

// WarningsOnUpdate returns warnings for the given update.
func (validatingAdmissionPolicyStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

// AllowUnconditionalUpdate is the default update policy for validatingAdmissionPolicy objects. Status update should
// only be allowed if version match.
func (validatingAdmissionPolicyStrategy) AllowUnconditionalUpdate() bool {
	return false
}
