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

package lease

import (
	"context"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/storage/names"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/coordination"
	"k8s.io/kubernetes/pkg/apis/coordination/validation"
	"k8s.io/kubernetes/pkg/features"
)

// leaseStrategy implements verification logic for Leases.
type leaseStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating Lease objects.
var Strategy = leaseStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

// NamespaceScoped returns true because all Lease' need to be within a namespace.
func (leaseStrategy) NamespaceScoped() bool {
	return true
}

// PrepareForCreate prepares Lease for creation.
func (leaseStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	lease := obj.(*coordination.Lease)
	if !utilfeature.DefaultFeatureGate.Enabled(features.CoordinatedLeaderElection) {
		lease.Spec.Strategy = nil
		lease.Spec.PreferredHolder = nil
	}

}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (leaseStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	oldLease := old.(*coordination.Lease)
	newLease := obj.(*coordination.Lease)
	if !utilfeature.DefaultFeatureGate.Enabled(features.CoordinatedLeaderElection) {
		if oldLease == nil || oldLease.Spec.Strategy == nil {
			newLease.Spec.Strategy = nil
		}
		if oldLease == nil || oldLease.Spec.PreferredHolder == nil {
			newLease.Spec.PreferredHolder = nil
		}
	}
}

// Validate validates a new Lease.
func (leaseStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	lease := obj.(*coordination.Lease)
	return validation.ValidateLease(lease)
}

// WarningsOnCreate returns warnings for the creation of the given object.
func (leaseStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string { return nil }

// Canonicalize normalizes the object after validation.
func (leaseStrategy) Canonicalize(obj runtime.Object) {
}

// AllowCreateOnUpdate is true for Lease; this means you may create one with a PUT request.
func (leaseStrategy) AllowCreateOnUpdate() bool {
	return true
}

// ValidateUpdate is the default update validation for an end user.
func (leaseStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateLeaseUpdate(obj.(*coordination.Lease), old.(*coordination.Lease))
}

// WarningsOnUpdate returns warnings for the given update.
func (leaseStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

// AllowUnconditionalUpdate is the default update policy for Lease objects.
func (leaseStrategy) AllowUnconditionalUpdate() bool {
	return false
}
