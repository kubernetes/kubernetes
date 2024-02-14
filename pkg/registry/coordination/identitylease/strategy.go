/*
Copyright 2018 The Kubernetes Authors.

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

package identitylease

import (
	"context"
	"fmt"

	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/coordination"
	"k8s.io/kubernetes/pkg/apis/coordination/validation"
)

// identityLeaseStrategy implements verification logic for identityLeases.
type identityLeaseStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating identityLease objects.
var Strategy = identityLeaseStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

// NamespaceScoped returns true because all identityLease' need to be within a namespace.
func (identityLeaseStrategy) NamespaceScoped() bool {
	return true
}

// PrepareForCreate prepares identityLease for creation.
func (identityLeaseStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (identityLeaseStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
}

// Validate validates a new identityLease.
func (identityLeaseStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	identityLease := obj.(*coordination.IdentityLease)
	return validation.ValidateIdentityLease(identityLease)
}

// WarningsOnCreate returns warnings for the creation of the given object.
func (identityLeaseStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

// Canonicalize normalizes the object after validation.
func (identityLeaseStrategy) Canonicalize(obj runtime.Object) {
}

// AllowCreateOnUpdate is true for identityLease; this means you may create one with a PUT request.
func (identityLeaseStrategy) AllowCreateOnUpdate() bool {
	return true
}

// ValidateUpdate is the default update validation for an end user.
func (identityLeaseStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateIdentityLeaseUpdate(obj.(*coordination.IdentityLease), old.(*coordination.IdentityLease))
}

// WarningsOnUpdate returns warnings for the given update.
func (identityLeaseStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

// AllowUnconditionalUpdate is the default update policy for identityLease objects.
func (identityLeaseStrategy) AllowUnconditionalUpdate() bool {
	return false
}

// GetAttrs returns labels and fields of a given object for filtering purposes.
func GetAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	identitylease, ok := obj.(*coordination.IdentityLease)
	if !ok {
		return nil, nil, fmt.Errorf("not a pod")
	}
	return labels.Set(identitylease.ObjectMeta.Labels), ToSelectableFields(identitylease), nil
}

// ToSelectableFields returns a field set that represents the object
// TODO: fields are not labels, and the validation rules for them do not apply.
func ToSelectableFields(identitylease *coordination.IdentityLease) fields.Set {
	objectMetaFieldsSet := generic.ObjectMetaFieldsSet(&identitylease.ObjectMeta, true)
	specificFieldsSet := fields.Set{
		"spec.canLeadLeas": string(identitylease.Spec.CanLeadLease),
	}
	return generic.MergeFieldsSets(objectMetaFieldsSet, specificFieldsSet)
}
