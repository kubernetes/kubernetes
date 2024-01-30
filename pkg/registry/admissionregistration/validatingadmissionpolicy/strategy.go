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

	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/admissionregistration"
	"k8s.io/kubernetes/pkg/apis/admissionregistration/validation"
	"k8s.io/kubernetes/pkg/registry/admissionregistration/resolver"
)

// validatingAdmissionPolicyStrategy implements verification logic for ValidatingAdmissionPolicy.
type validatingAdmissionPolicyStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
	authorizer       authorizer.Authorizer
	resourceResolver resolver.ResourceResolver
}

// NewStrategy is the default logic that applies when creating and updating validatingAdmissionPolicy objects.
func NewStrategy(authorizer authorizer.Authorizer, resourceResolver resolver.ResourceResolver) *validatingAdmissionPolicyStrategy {
	return &validatingAdmissionPolicyStrategy{
		ObjectTyper:      legacyscheme.Scheme,
		NameGenerator:    names.SimpleNameGenerator,
		authorizer:       authorizer,
		resourceResolver: resourceResolver,
	}
}

// NamespaceScoped returns false because ValidatingAdmissionPolicy is cluster-scoped resource.
func (v *validatingAdmissionPolicyStrategy) NamespaceScoped() bool {
	return false
}

// PrepareForCreate clears the status of an validatingAdmissionPolicy before creation.
func (v *validatingAdmissionPolicyStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	ic := obj.(*admissionregistration.ValidatingAdmissionPolicy)
	ic.Status = admissionregistration.ValidatingAdmissionPolicyStatus{}
	ic.Generation = 1
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (v *validatingAdmissionPolicyStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newIC := obj.(*admissionregistration.ValidatingAdmissionPolicy)
	oldIC := old.(*admissionregistration.ValidatingAdmissionPolicy)

	// Prevent any update on the Status object
	newIC.Status = oldIC.Status

	// Any changes to the spec increment the generation number, any changes to the
	// status should reflect the generation number of the corresponding object.
	// See metav1.ObjectMeta description for more information on Generation.
	if !apiequality.Semantic.DeepEqual(oldIC.Spec, newIC.Spec) {
		newIC.Generation = oldIC.Generation + 1
	}
}

// Validate validates a new validatingAdmissionPolicy.
func (v *validatingAdmissionPolicyStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	errs := validation.ValidateValidatingAdmissionPolicy(obj.(*admissionregistration.ValidatingAdmissionPolicy))
	if len(errs) == 0 {
		// if the object is well-formed, also authorize the paramKind
		if err := v.authorizeCreate(ctx, obj); err != nil {
			errs = append(errs, field.Forbidden(field.NewPath("spec", "paramKind"), err.Error()))
		}
	}
	return errs
}

// WarningsOnCreate returns warnings for the creation of the given object.
func (v *validatingAdmissionPolicyStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

// Canonicalize normalizes the object after validation.
func (v *validatingAdmissionPolicyStrategy) Canonicalize(obj runtime.Object) {
}

// AllowCreateOnUpdate is true for validatingAdmissionPolicy; this means you may create one with a PUT request.
func (v *validatingAdmissionPolicyStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (v *validatingAdmissionPolicyStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	errs := validation.ValidateValidatingAdmissionPolicyUpdate(obj.(*admissionregistration.ValidatingAdmissionPolicy), old.(*admissionregistration.ValidatingAdmissionPolicy))
	if len(errs) == 0 {
		// if the object is well-formed, also authorize the paramKind
		if err := v.authorizeUpdate(ctx, obj, old); err != nil {
			errs = append(errs, field.Forbidden(field.NewPath("spec", "paramKind"), err.Error()))
		}
	}
	return errs
}

// WarningsOnUpdate returns warnings for the given update.
func (v *validatingAdmissionPolicyStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

// AllowUnconditionalUpdate is the default update policy for validatingAdmissionPolicy objects. Status update should
// only be allowed if version match.
func (v *validatingAdmissionPolicyStrategy) AllowUnconditionalUpdate() bool {
	return false
}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (v *validatingAdmissionPolicyStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"admissionregistration.k8s.io/v1alpha1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
		"admissionregistration.k8s.io/v1beta1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
	}

	return fields
}

type validatingAdmissionPolicyStatusStrategy struct {
	*validatingAdmissionPolicyStrategy
}

// ValidateUpdate is the default update validation of an update to the status object.
func (s *validatingAdmissionPolicyStatusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateValidatingAdmissionPolicyStatusUpdate(obj.(*admissionregistration.ValidatingAdmissionPolicy), old.(*admissionregistration.ValidatingAdmissionPolicy))
}

// PrepareForUpdate differs from the main strategy where setting the spec is not
// allowed, but setting status is OK.
func (s *validatingAdmissionPolicyStatusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newIC := obj.(*admissionregistration.ValidatingAdmissionPolicy)
	oldIC := old.(*admissionregistration.ValidatingAdmissionPolicy)

	// Prevent any update on the Spec object from Status Strategy
	newIC.Spec = oldIC.Spec

	metav1.ResetObjectMetaForStatus(&newIC.ObjectMeta, &oldIC.ObjectMeta)
	// No change in the generation.
}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (s *validatingAdmissionPolicyStatusStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	return map[fieldpath.APIVersion]*fieldpath.Set{
		"admissionregistration.k8s.io/v1alpha1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("spec"),
			fieldpath.MakePathOrDie("metadata"),
		),
		"admissionregistration.k8s.io/v1beta1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("spec"),
			fieldpath.MakePathOrDie("metadata"),
		),
	}
}

// NewStatusStrategy creates a strategy for operating the status object.
func NewStatusStrategy(policyStrategy *validatingAdmissionPolicyStrategy) *validatingAdmissionPolicyStatusStrategy {
	return &validatingAdmissionPolicyStatusStrategy{validatingAdmissionPolicyStrategy: policyStrategy}
}
