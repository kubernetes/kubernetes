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

package mutatingadmissionpolicybinding

import (
	"context"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/admissionregistration"
	"k8s.io/kubernetes/pkg/apis/admissionregistration/validation"
	"k8s.io/kubernetes/pkg/registry/admissionregistration/resolver"
)

// MutatingAdmissionPolicyBindingStrategy implements verification logic for MutatingAdmissionPolicyBinding.
type mutatingAdmissionPolicyBindingStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
	authorizer       authorizer.Authorizer
	policyGetter     PolicyGetter
	resourceResolver resolver.ResourceResolver
}

type PolicyGetter interface {
	// GetMutatingAdmissionPolicy returns a GetMutatingAdmissionPolicy
	// by its name. There is no namespace because it is cluster-scoped.
	GetMutatingAdmissionPolicy(ctx context.Context, name string) (*admissionregistration.MutatingAdmissionPolicy, error)
}

// NewStrategy is the default logic that applies when creating and updating MutatingAdmissionPolicyBinding objects.
func NewStrategy(authorizer authorizer.Authorizer, policyGetter PolicyGetter, resourceResolver resolver.ResourceResolver) *mutatingAdmissionPolicyBindingStrategy {
	return &mutatingAdmissionPolicyBindingStrategy{
		ObjectTyper:      legacyscheme.Scheme,
		NameGenerator:    names.SimpleNameGenerator,
		authorizer:       authorizer,
		policyGetter:     policyGetter,
		resourceResolver: resourceResolver,
	}
}

// NamespaceScoped returns false because MutatingAdmissionPolicyBinding is cluster-scoped resource.
func (v *mutatingAdmissionPolicyBindingStrategy) NamespaceScoped() bool {
	return false
}

// PrepareForCreate clears the status of an MutatingAdmissionPolicyBinding before creation.
func (v *mutatingAdmissionPolicyBindingStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	ic := obj.(*admissionregistration.MutatingAdmissionPolicyBinding)
	ic.Generation = 1
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (v *mutatingAdmissionPolicyBindingStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newIC := obj.(*admissionregistration.MutatingAdmissionPolicyBinding)
	oldIC := old.(*admissionregistration.MutatingAdmissionPolicyBinding)

	// Any changes to the spec increment the generation number, any changes to the
	// status should reflect the generation number of the corresponding object.
	// See metav1.ObjectMeta description for more information on Generation.
	if !apiequality.Semantic.DeepEqual(oldIC.Spec, newIC.Spec) {
		newIC.Generation = oldIC.Generation + 1
	}
}

// Validate validates a new MutatingAdmissionPolicyBinding.
func (v *mutatingAdmissionPolicyBindingStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	errs := validation.ValidateMutatingAdmissionPolicyBinding(obj.(*admissionregistration.MutatingAdmissionPolicyBinding))
	if len(errs) == 0 {
		// if the object is well-formed, also authorize the paramRef
		if err := v.authorizeCreate(ctx, obj); err != nil {
			errs = append(errs, field.Forbidden(field.NewPath("spec", "paramRef"), err.Error()))
		}
	}
	return errs
}

// WarningsOnCreate returns warnings for the creation of the given object.
func (v *mutatingAdmissionPolicyBindingStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

// Canonicalize normalizes the object after validation.
func (v *mutatingAdmissionPolicyBindingStrategy) Canonicalize(obj runtime.Object) {
}

// AllowCreateOnUpdate is false for MutatingAdmissionPolicyBinding; this means you may not create one with a PUT request.
func (v *mutatingAdmissionPolicyBindingStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (v *mutatingAdmissionPolicyBindingStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	errs := validation.ValidateMutatingAdmissionPolicyBindingUpdate(obj.(*admissionregistration.MutatingAdmissionPolicyBinding), old.(*admissionregistration.MutatingAdmissionPolicyBinding))
	if len(errs) == 0 {
		// if the object is well-formed, also authorize the paramRef
		if err := v.authorizeUpdate(ctx, obj, old); err != nil {
			errs = append(errs, field.Forbidden(field.NewPath("spec", "paramRef"), err.Error()))
		}
	}
	return errs
}

// WarningsOnUpdate returns warnings for the given update.
func (v *mutatingAdmissionPolicyBindingStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

// AllowUnconditionalUpdate is the default update policy for MutatingAdmissionPolicyBinding objects. Status update should
// only be allowed if version match.
func (v *mutatingAdmissionPolicyBindingStrategy) AllowUnconditionalUpdate() bool {
	return false
}
