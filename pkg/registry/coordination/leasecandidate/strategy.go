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

package leasecandidate

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

// LeaseCandidateStrategy implements verification logic for leasecandidates.
type LeaseCandidateStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating leasecandidate objects.
var Strategy = LeaseCandidateStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

// NamespaceScoped returns true because all leasecandidate' need to be within a namespace.
func (LeaseCandidateStrategy) NamespaceScoped() bool {
	return true
}

// PrepareForCreate prepares leasecandidate for creation.
func (LeaseCandidateStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (LeaseCandidateStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
}

// Validate validates a new leasecandidate.
func (LeaseCandidateStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	leasecandidate := obj.(*coordination.LeaseCandidate)
	return validation.ValidateLeaseCandidate(leasecandidate)
}

// WarningsOnCreate returns warnings for the creation of the given object.
func (LeaseCandidateStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

// Canonicalize normalizes the object after validation.
func (LeaseCandidateStrategy) Canonicalize(obj runtime.Object) {
}

// AllowCreateOnUpdate is true for leasecandidate; this means you may create one with a PUT request.
func (LeaseCandidateStrategy) AllowCreateOnUpdate() bool {
	return true
}

// ValidateUpdate is the default update validation for an end user.
func (LeaseCandidateStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateLeaseCandidateUpdate(obj.(*coordination.LeaseCandidate), old.(*coordination.LeaseCandidate))
}

// WarningsOnUpdate returns warnings for the given update.
func (LeaseCandidateStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

// AllowUnconditionalUpdate is the default update policy for leasecandidate objects.
func (LeaseCandidateStrategy) AllowUnconditionalUpdate() bool {
	return false
}

// GetAttrs returns labels and fields of a given object for filtering purposes.
func GetAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	leasecandidate, ok := obj.(*coordination.LeaseCandidate)
	if !ok {
		return nil, nil, fmt.Errorf("not a pod")
	}
	return labels.Set(leasecandidate.ObjectMeta.Labels), ToSelectableFields(leasecandidate), nil
}

// ToSelectableFields returns a field set that represents the object
// TODO: fields are not labels, and the validation rules for them do not apply.
func ToSelectableFields(leasecandidate *coordination.LeaseCandidate) fields.Set {
	objectMetaFieldsSet := generic.ObjectMetaFieldsSet(&leasecandidate.ObjectMeta, true)
	specificFieldsSet := fields.Set{
		"spec.canLeadLeas": string(leasecandidate.Spec.TargetLease),
	}
	return generic.MergeFieldsSets(objectMetaFieldsSet, specificFieldsSet)
}
