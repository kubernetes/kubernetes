/*
Copyright 2023 The Kubernetes Authors.

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

package servicecidr

import (
	"context"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/networking"
	"k8s.io/kubernetes/pkg/apis/networking/validation"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
)

// serviceCIDRStrategy implements verification logic for ServiceCIDR allocators.
type serviceCIDRStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating Replication ServiceCIDR objects.
var Strategy = serviceCIDRStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

// Strategy should implement rest.RESTCreateStrategy
var _ rest.RESTCreateStrategy = Strategy

// Strategy should implement rest.RESTUpdateStrategy
var _ rest.RESTUpdateStrategy = Strategy

// NamespaceScoped returns false because all ServiceCIDRes is cluster scoped.
func (serviceCIDRStrategy) NamespaceScoped() bool {
	return false
}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (serviceCIDRStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"networking/v1alpha1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
	}
	return fields
}

// PrepareForCreate clears the status of an ServiceCIDR before creation.
func (serviceCIDRStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	_ = obj.(*networking.ServiceCIDR)

}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (serviceCIDRStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newServiceCIDR := obj.(*networking.ServiceCIDR)
	oldServiceCIDR := old.(*networking.ServiceCIDR)

	_, _ = newServiceCIDR, oldServiceCIDR
}

// Validate validates a new ServiceCIDR.
func (serviceCIDRStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	cidrConfig := obj.(*networking.ServiceCIDR)
	err := validation.ValidateServiceCIDR(cidrConfig)
	return err
}

// Canonicalize normalizes the object after validation.
func (serviceCIDRStrategy) Canonicalize(obj runtime.Object) {
}

// AllowCreateOnUpdate is false for ServiceCIDR; this means POST is needed to create one.
func (serviceCIDRStrategy) AllowCreateOnUpdate() bool {
	return false
}

// WarningsOnCreate returns warnings for the creation of the given object.
func (serviceCIDRStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

// ValidateUpdate is the default update validation for an end user.
func (serviceCIDRStrategy) ValidateUpdate(ctx context.Context, new, old runtime.Object) field.ErrorList {
	newServiceCIDR := new.(*networking.ServiceCIDR)
	oldServiceCIDR := old.(*networking.ServiceCIDR)
	errList := validation.ValidateServiceCIDR(newServiceCIDR)
	errList = append(errList, validation.ValidateServiceCIDRUpdate(newServiceCIDR, oldServiceCIDR)...)
	return errList
}

// AllowUnconditionalUpdate is the default update policy for ServiceCIDR objects.
func (serviceCIDRStrategy) AllowUnconditionalUpdate() bool {
	return true
}

// WarningsOnUpdate returns warnings for the given update.
func (serviceCIDRStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

type serviceCIDRStatusStrategy struct {
	serviceCIDRStrategy
}

// StatusStrategy implements logic used to validate and prepare for updates of the status subresource
var StatusStrategy = serviceCIDRStatusStrategy{Strategy}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (serviceCIDRStatusStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"networking/v1alpha1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("spec"),
		),
	}
	return fields
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update of status
func (serviceCIDRStatusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newServiceCIDR := obj.(*networking.ServiceCIDR)
	oldServiceCIDR := old.(*networking.ServiceCIDR)
	// status changes are not allowed to update spec
	newServiceCIDR.Spec = oldServiceCIDR.Spec
}

// ValidateUpdate is the default update validation for an end user updating status
func (serviceCIDRStatusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateServiceCIDRStatusUpdate(obj.(*networking.ServiceCIDR), old.(*networking.ServiceCIDR))
}

// WarningsOnUpdate returns warnings for the given update.
func (serviceCIDRStatusStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}
