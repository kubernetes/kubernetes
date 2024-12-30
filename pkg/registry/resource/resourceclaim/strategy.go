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

package resourceclaim

import (
	"context"
	"errors"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/names"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/kubernetes/pkg/apis/resource/validation"
	"k8s.io/kubernetes/pkg/features"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
)

const (
	DRAAdminNamespaceLabel = "kubernetes.io/dra-admin-access"
)

type NamespaceGetter interface {
	Get(ctx context.Context, name string, options *metav1.GetOptions) (runtime.Object, error)
}

// resourceclaimStrategy implements behavior for ResourceClaim objects
type resourceclaimStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
	namespaceRest NamespaceGetter
}

// Strategy is the default logic that applies when creating and updating
// ResourceClaim objects via the REST API.
var Strategy = resourceclaimStrategy{legacyscheme.Scheme, names.SimpleNameGenerator, nil}

// SetNamespaceStore sets the namespace store rest object for the strategy.
func (s *resourceclaimStrategy) SetNamespaceStore(rest NamespaceGetter) {
	s.namespaceRest = rest
}

func (resourceclaimStrategy) NamespaceScoped() bool {
	return true
}

// GetResetFields returns the set of fields that get reset by the strategy and
// should not be modified by the user. For a new ResourceClaim that is the
// status.
func (resourceclaimStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"resource.k8s.io/v1alpha3": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
		"resource.k8s.io/v1beta1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
	}

	return fields
}

func (resourceclaimStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	claim := obj.(*resource.ResourceClaim)
	// Status must not be set by user on create.
	claim.Status = resource.ResourceClaimStatus{}

	dropDisabledFields(claim, nil)
}

func (s resourceclaimStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	claim := obj.(*resource.ResourceClaim)

	allErrs := authorizedForAdmin(ctx, claim, s.namespaceRest)
	return append(allErrs, validation.ValidateResourceClaim(claim)...)
}

func (resourceclaimStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

func (resourceclaimStrategy) Canonicalize(obj runtime.Object) {
}

func (resourceclaimStrategy) AllowCreateOnUpdate() bool {
	return false
}

func (resourceclaimStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newClaim := obj.(*resource.ResourceClaim)
	oldClaim := old.(*resource.ResourceClaim)
	newClaim.Status = oldClaim.Status

	dropDisabledFields(newClaim, oldClaim)
}

func (s resourceclaimStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	newClaim := obj.(*resource.ResourceClaim)
	oldClaim := old.(*resource.ResourceClaim)
	allErrs := authorizedForAdmin(ctx, newClaim, s.namespaceRest)
	allErrs = append(allErrs, validation.ValidateResourceClaim(newClaim)...)
	return append(allErrs, validation.ValidateResourceClaimUpdate(newClaim, oldClaim)...)
}

func (resourceclaimStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func (resourceclaimStrategy) AllowUnconditionalUpdate() bool {
	return true
}

type resourceclaimStatusStrategy struct {
	resourceclaimStrategy
}

var StatusStrategy = resourceclaimStatusStrategy{Strategy}

// GetResetFields returns the set of fields that get reset by the strategy and
// should not be modified by the user. For a status update that is the spec.
func (resourceclaimStatusStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"resource.k8s.io/v1alpha3": fieldpath.NewSet(
			fieldpath.MakePathOrDie("spec"),
		),
		"resource.k8s.io/v1beta1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("spec"),
		),
	}

	return fields
}

func (resourceclaimStatusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newClaim := obj.(*resource.ResourceClaim)
	oldClaim := old.(*resource.ResourceClaim)
	newClaim.Spec = oldClaim.Spec
	metav1.ResetObjectMetaForStatus(&newClaim.ObjectMeta, &oldClaim.ObjectMeta)

	dropDeallocatedStatusDevices(newClaim, oldClaim)
	dropDisabledFields(newClaim, oldClaim)
}

func (resourceclaimStatusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	newClaim := obj.(*resource.ResourceClaim)
	oldClaim := old.(*resource.ResourceClaim)
	return validation.ValidateResourceClaimStatusUpdate(newClaim, oldClaim)
}

// WarningsOnUpdate returns warnings for the given update.
func (resourceclaimStatusStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

// Match returns a generic matcher for a given label and field selector.
func Match(label labels.Selector, field fields.Selector) storage.SelectionPredicate {
	return storage.SelectionPredicate{
		Label:    label,
		Field:    field,
		GetAttrs: GetAttrs,
	}
}

// GetAttrs returns labels and fields of a given object for filtering purposes.
func GetAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	claim, ok := obj.(*resource.ResourceClaim)
	if !ok {
		return nil, nil, errors.New("not a resourceclaim")
	}
	return labels.Set(claim.Labels), toSelectableFields(claim), nil
}

// toSelectableFields returns a field set that represents the object
func toSelectableFields(claim *resource.ResourceClaim) fields.Set {
	fields := generic.ObjectMetaFieldsSet(&claim.ObjectMeta, true)
	return fields
}

// authorizedForAdmin checks if the request is authorized to get admin access to devices
// based on namespace label
func authorizedForAdmin(ctx context.Context, newClaim *resource.ResourceClaim, namespaceRest NamespaceGetter) field.ErrorList {
	allErrs := field.ErrorList{}
	adminRequested := false

	if !utilfeature.DefaultFeatureGate.Enabled(features.DRAAdminAccess) {
		// No need to validate unless feature gate is enabled
		return allErrs
	}

	for i := range newClaim.Spec.Devices.Requests {
		value := newClaim.Spec.Devices.Requests[i].AdminAccess
		if value != nil && *value {
			adminRequested = true
			break
		}
	}
	if !adminRequested {
		// No need to validate unless admin access is requested
		return allErrs
	}
	if namespaceRest == nil {
		return append(allErrs, field.Forbidden(field.NewPath(""), "namespace store is nil"))
	}

	namespaceName := newClaim.Namespace
	// Retrieve the namespace object from the store
	obj, err := namespaceRest.Get(ctx, namespaceName, &metav1.GetOptions{ResourceVersion: "0"})
	if err != nil {
		return append(allErrs, field.Forbidden(field.NewPath(""), "namespace object cannot be retrieved"))
	}
	ns, ok := obj.(*core.Namespace)
	if !ok {
		return append(allErrs, field.Forbidden(field.NewPath(""), "namespace object is not of type core.Namespace"))
	}
	if value, exists := ns.Labels[DRAAdminNamespaceLabel]; !(exists && value == "true") {
		return append(allErrs, field.Forbidden(field.NewPath(""), "admin access to devices is not allowed in namespace without DRA Admin Access label"))
	}

	return allErrs
}

// dropDisabledFields removes fields which are covered by a feature gate.
func dropDisabledFields(newClaim, oldClaim *resource.ResourceClaim) {
	dropDisabledDRAAdminAccessFields(newClaim, oldClaim)
	dropDisabledDRAResourceClaimDeviceStatusFields(newClaim, oldClaim)
}

func dropDisabledDRAAdminAccessFields(newClaim, oldClaim *resource.ResourceClaim) {
	if utilfeature.DefaultFeatureGate.Enabled(features.DRAAdminAccess) {
		// No need to drop anything.
		return
	}
	if draAdminAccessFeatureInUse(oldClaim) {
		// If anything was set in the past, then fields must not get
		// dropped on potentially unrelated updates and, for example,
		// adding a status with AdminAccess=true is allowed. The
		// scheduler typically doesn't do that (it also checks the
		// feature gate and refuses to schedule), but the apiserver
		// would allow it.
		return
	}

	for i := range newClaim.Spec.Devices.Requests {
		newClaim.Spec.Devices.Requests[i].AdminAccess = nil
	}

	if newClaim.Status.Allocation == nil {
		return
	}
	for i := range newClaim.Status.Allocation.Devices.Results {
		newClaim.Status.Allocation.Devices.Results[i].AdminAccess = nil
	}
}

func draAdminAccessFeatureInUse(claim *resource.ResourceClaim) bool {
	if claim == nil {
		return false
	}

	for _, request := range claim.Spec.Devices.Requests {
		if request.AdminAccess != nil {
			return true
		}
	}

	if allocation := claim.Status.Allocation; allocation != nil {
		for _, result := range allocation.Devices.Results {
			if result.AdminAccess != nil {
				return true
			}
		}
	}

	return false
}

func dropDisabledDRAResourceClaimDeviceStatusFields(newClaim, oldClaim *resource.ResourceClaim) {
	isDRAResourceClaimDeviceStatusInUse := (oldClaim != nil && len(oldClaim.Status.Devices) > 0)
	// drop resourceClaim.Status.Devices field if feature gate is not enabled and it was not in use
	if !utilfeature.DefaultFeatureGate.Enabled(features.DRAResourceClaimDeviceStatus) && !isDRAResourceClaimDeviceStatusInUse {
		newClaim.Status.Devices = nil
	}
}

// dropDeallocatedStatusDevices removes the status.devices that were allocated
// in the oldClaim and that have been removed in the newClaim.
func dropDeallocatedStatusDevices(newClaim, oldClaim *resource.ResourceClaim) {
	isDRAResourceClaimDeviceStatusInUse := (oldClaim != nil && len(oldClaim.Status.Devices) > 0)
	if !utilfeature.DefaultFeatureGate.Enabled(features.DRAResourceClaimDeviceStatus) && !isDRAResourceClaimDeviceStatusInUse {
		return
	}

	deallocatedDevices := sets.New[structured.DeviceID]()

	if oldClaim.Status.Allocation != nil {
		// Get all devices in the oldClaim.
		for _, result := range oldClaim.Status.Allocation.Devices.Results {
			deviceID := structured.MakeDeviceID(result.Driver, result.Pool, result.Device)
			deallocatedDevices.Insert(deviceID)
		}
	}

	// Remove devices from deallocatedDevices that are still in newClaim.
	if newClaim.Status.Allocation != nil {
		for _, result := range newClaim.Status.Allocation.Devices.Results {
			deviceID := structured.MakeDeviceID(result.Driver, result.Pool, result.Device)
			deallocatedDevices.Delete(deviceID)
		}
	}

	// Remove from newClaim.Status.Devices.
	n := 0
	for _, device := range newClaim.Status.Devices {
		deviceID := structured.MakeDeviceID(device.Driver, device.Pool, device.Device)
		if !deallocatedDevices.Has(deviceID) {
			newClaim.Status.Devices[n] = device
			n++
		}
	}
	newClaim.Status.Devices = newClaim.Status.Devices[:n]

	if len(newClaim.Status.Devices) == 0 {
		newClaim.Status.Devices = nil
	}
}
