/*
Copyright 2017 The Kubernetes Authors.

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

package networkpolicy

import (
	"context"
	"reflect"

	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/storage/names"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/networking"
	"k8s.io/kubernetes/pkg/apis/networking/validation"
	"k8s.io/kubernetes/pkg/features"
)

// networkPolicyStrategy implements verification logic for NetworkPolicies
type networkPolicyStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating NetworkPolicy objects.
var Strategy = networkPolicyStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

// NamespaceScoped returns true because all NetworkPolicies need to be within a namespace.
func (networkPolicyStrategy) NamespaceScoped() bool {
	return true
}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (networkPolicyStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"extensions/v1beta1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
		"networking.k8s.io/v1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
	}
	return fields
}

// PrepareForCreate clears the status of a NetworkPolicy before creation.
func (networkPolicyStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	networkPolicy := obj.(*networking.NetworkPolicy)

	if utilfeature.DefaultFeatureGate.Enabled(features.NetworkPolicyStatus) {
		// Create does not set a status when operation is not directed to status subresource
		networkPolicy.Status = networking.NetworkPolicyStatus{}
	}

	networkPolicy.Generation = 1

}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (networkPolicyStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newNetworkPolicy := obj.(*networking.NetworkPolicy)
	oldNetworkPolicy := old.(*networking.NetworkPolicy)

	// We copy the status if the FG is enabled, or if previously there was already data on the conditions field
	// As soon as the FeatureGate is removed, the whole if statement should be removed as well
	if utilfeature.DefaultFeatureGate.Enabled(features.NetworkPolicyStatus) || len(oldNetworkPolicy.Status.Conditions) > 0 {
		// Update is not allowed to set status when the operation is not directed to status subresource
		newNetworkPolicy.Status = oldNetworkPolicy.Status
	}

	// Any changes to the spec increment the generation number, any changes to the
	// status should reflect the generation number of the corresponding object.
	// See metav1.ObjectMeta description for more information on Generation.
	if !reflect.DeepEqual(oldNetworkPolicy.Spec, newNetworkPolicy.Spec) {
		newNetworkPolicy.Generation = oldNetworkPolicy.Generation + 1
	}
}

// Validate validates a new NetworkPolicy.
func (networkPolicyStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	networkPolicy := obj.(*networking.NetworkPolicy)
	return validation.ValidateNetworkPolicy(networkPolicy)
}

// WarningsOnCreate returns warnings for the creation of the given object.
func (networkPolicyStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

// Canonicalize normalizes the object after validation.
func (networkPolicyStrategy) Canonicalize(obj runtime.Object) {}

// AllowCreateOnUpdate is false for NetworkPolicy; this means POST is needed to create one.
func (networkPolicyStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (networkPolicyStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	validationErrorList := validation.ValidateNetworkPolicy(obj.(*networking.NetworkPolicy))
	updateErrorList := validation.ValidateNetworkPolicyUpdate(obj.(*networking.NetworkPolicy), old.(*networking.NetworkPolicy))
	return append(validationErrorList, updateErrorList...)
}

// WarningsOnUpdate returns warnings for the given update.
func (networkPolicyStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

// AllowUnconditionalUpdate is the default update policy for NetworkPolicy objects.
func (networkPolicyStrategy) AllowUnconditionalUpdate() bool {
	return true
}

type networkPolicyStatusStrategy struct {
	networkPolicyStrategy
}

// StatusStrategy implements logic used to validate and prepare for updates of the status subresource
var StatusStrategy = networkPolicyStatusStrategy{Strategy}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (networkPolicyStatusStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"extensions/v1beta1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("spec"),
		),
		"networking.k8s.io/v1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("spec"),
		),
	}
	return fields
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update of status
func (networkPolicyStatusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newNetworkPolicy := obj.(*networking.NetworkPolicy)
	oldNetworkPolicy := old.(*networking.NetworkPolicy)
	// status changes are not allowed to update spec
	newNetworkPolicy.Spec = oldNetworkPolicy.Spec

	if !utilfeature.DefaultFeatureGate.Enabled(features.NetworkPolicyStatus) {
		// As network policy status is composed only of an array of conditions, we can say that the status
		// is in use if the condition array is bigger than 0.
		// quoting @thockin: "we generally keep data in this case, but no updates except to clear it"
		if len(newNetworkPolicy.Status.Conditions) == 0 {
			newNetworkPolicy.Status = networking.NetworkPolicyStatus{}
		} else {
			// keep the old status in case of the update is not to clear it
			newNetworkPolicy.Status = oldNetworkPolicy.Status
		}
	}
}

// ValidateUpdate is the default update validation for an end user updating status
func (networkPolicyStatusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateNetworkPolicyStatusUpdate(obj.(*networking.NetworkPolicy).Status,
		old.(*networking.NetworkPolicy).Status, field.NewPath("status"))
}

// WarningsOnUpdate returns warnings for the given update.
func (networkPolicyStatusStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}
