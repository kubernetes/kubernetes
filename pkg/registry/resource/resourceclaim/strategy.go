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

	"sigs.k8s.io/structured-merge-diff/v6/fieldpath"

	"k8s.io/apimachinery/pkg/api/operation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/names"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	v1 "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/api/resourceclaimspec"
	"k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/kubernetes/pkg/apis/resource/validation"
	"k8s.io/kubernetes/pkg/features"
	resourceutils "k8s.io/kubernetes/pkg/registry/resource"
	"k8s.io/utils/ptr"
)

// resourceclaimStrategy implements behavior for ResourceClaim objects
type resourceclaimStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
	nsClient v1.NamespaceInterface
}

// NewStrategy is the default logic that applies when creating and updating ResourceClaim objects.
func NewStrategy(nsClient v1.NamespaceInterface) *resourceclaimStrategy {
	return &resourceclaimStrategy{
		legacyscheme.Scheme,
		names.SimpleNameGenerator,
		nsClient,
	}
}

func (*resourceclaimStrategy) NamespaceScoped() bool {
	return true
}

// GetResetFields returns the set of fields that get reset by the strategy and
// should not be modified by the user. For a new ResourceClaim that is the
// status.
func (*resourceclaimStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"resource.k8s.io/v1alpha3": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
		"resource.k8s.io/v1beta1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
		"resource.k8s.io/v1beta2": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
		"resource.k8s.io/v1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
	}

	return fields
}

func (*resourceclaimStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	claim := obj.(*resource.ResourceClaim)
	// Status must not be set by user on create.
	claim.Status = resource.ResourceClaimStatus{}

	dropDisabledSpecFields(claim, nil)
}

func (s *resourceclaimStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	claim := obj.(*resource.ResourceClaim)

	allErrs := resourceutils.AuthorizedForAdmin(ctx, claim.Spec.Devices.Requests, claim.Namespace, s.nsClient)
	allErrs = append(allErrs, validation.ValidateResourceClaim(claim)...)
	return rest.ValidateDeclarativelyWithMigrationChecks(ctx, legacyscheme.Scheme, claim, nil, allErrs, operation.Create, rest.WithNormalizationRules(validation.ResourceNormalizationRules))
}

func (*resourceclaimStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

func (*resourceclaimStrategy) Canonicalize(obj runtime.Object) {
}

func (*resourceclaimStrategy) AllowCreateOnUpdate() bool {
	return false
}

func (*resourceclaimStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newClaim := obj.(*resource.ResourceClaim)
	oldClaim := old.(*resource.ResourceClaim)
	newClaim.Status = oldClaim.Status

	dropDisabledSpecFields(newClaim, oldClaim)
}

func (s *resourceclaimStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	newClaim := obj.(*resource.ResourceClaim)
	oldClaim := old.(*resource.ResourceClaim)
	// AuthorizedForAdmin isn't needed here because the spec is immutable.
	errorList := validation.ValidateResourceClaimUpdate(newClaim, oldClaim)
	return rest.ValidateDeclarativelyWithMigrationChecks(ctx, legacyscheme.Scheme, newClaim, oldClaim, errorList, operation.Update, rest.WithNormalizationRules(validation.ResourceNormalizationRules))
}

func (*resourceclaimStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func (*resourceclaimStrategy) AllowUnconditionalUpdate() bool {
	return true
}

type resourceclaimStatusStrategy struct {
	*resourceclaimStrategy
}

// NewStatusStrategy creates a strategy for operating the status object.
func NewStatusStrategy(resourceclaimStrategy *resourceclaimStrategy) *resourceclaimStatusStrategy {
	return &resourceclaimStatusStrategy{resourceclaimStrategy}
}

// GetResetFields returns the set of fields that get reset by the strategy and
// should not be modified by the user. For a status update that is the spec.
func (*resourceclaimStatusStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"resource.k8s.io/v1alpha3": fieldpath.NewSet(
			fieldpath.MakePathOrDie("metadata"),
			fieldpath.MakePathOrDie("spec"),
		),
		"resource.k8s.io/v1beta1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("metadata"),
			fieldpath.MakePathOrDie("spec"),
		),
		"resource.k8s.io/v1beta2": fieldpath.NewSet(
			fieldpath.MakePathOrDie("metadata"),
			fieldpath.MakePathOrDie("spec"),
		),
		"resource.k8s.io/v1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("metadata"),
			fieldpath.MakePathOrDie("spec"),
		),
	}

	return fields
}

func (*resourceclaimStatusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newClaim := obj.(*resource.ResourceClaim)
	oldClaim := old.(*resource.ResourceClaim)
	newClaim.Spec = oldClaim.Spec
	metav1.ResetObjectMetaForStatus(&newClaim.ObjectMeta, &oldClaim.ObjectMeta)

	dropDisabledStatusFields(newClaim, oldClaim)
	dropDeallocatedStatusDevices(newClaim, oldClaim) // NOP if fields got dropped, so do this last.
}

func (r *resourceclaimStatusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	newClaim := obj.(*resource.ResourceClaim)
	oldClaim := old.(*resource.ResourceClaim)
	var newAllocationResult, oldAllocationResult []resource.DeviceRequestAllocationResult
	if newClaim.Status.Allocation != nil {
		newAllocationResult = newClaim.Status.Allocation.Devices.Results
	}
	if oldClaim.Status.Allocation != nil {
		oldAllocationResult = oldClaim.Status.Allocation.Devices.Results
	}
	errs := resourceutils.AuthorizedForAdminStatus(ctx, newAllocationResult, oldAllocationResult, newClaim.Namespace, r.nsClient)
	errs = append(errs, validation.ValidateResourceClaimStatusUpdate(newClaim, oldClaim)...)
	return rest.ValidateDeclarativelyWithMigrationChecks(ctx, legacyscheme.Scheme, newClaim, oldClaim, errs, operation.Update)
}

// WarningsOnUpdate returns warnings for the given update.
func (*resourceclaimStatusStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
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

// dropDisabledSpecFields removes fields from the spec which are covered by a feature gate.
func dropDisabledSpecFields(newClaim, oldClaim *resource.ResourceClaim) {
	var oldClaimSpec *resource.ResourceClaimSpec
	if oldClaim != nil {
		oldClaimSpec = &oldClaim.Spec
	}
	resourceclaimspec.DropDisabledFields(&newClaim.Spec, oldClaimSpec)
}

// dropDisabledStatusFields removes fields from the status which are covered by a feature gate.
func dropDisabledStatusFields(newClaim, oldClaim *resource.ResourceClaim) {
	dropDisabledDRAResourceClaimDeviceStatusFields(newClaim, oldClaim)
	dropDisabledDRAAdminAccessStatusFields(newClaim, oldClaim)
	dropDisabledDRAResourceClaimConsumableCapacityStatusFields(newClaim, oldClaim)
	dropDeviceBindingConditionsFields(newClaim, oldClaim)
}

func dropDisabledDRAAdminAccessStatusFields(newClaim, oldClaim *resource.ResourceClaim) {
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

	if resourceclaimspec.DRAAdminAccessFeatureInUse(&claim.Spec) {
		return true
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

func isDRAResourceClaimDeviceStatusInUse(claim *resource.ResourceClaim) bool {
	return claim != nil && len(claim.Status.Devices) > 0
}

func dropDisabledDRAResourceClaimDeviceStatusFields(newClaim, oldClaim *resource.ResourceClaim) {
	// drop resourceClaim.Status.Devices field if feature gate is not enabled and it was not in use
	if !utilfeature.DefaultFeatureGate.Enabled(features.DRAResourceClaimDeviceStatus) && !isDRAResourceClaimDeviceStatusInUse(oldClaim) {
		newClaim.Status.Devices = nil
	}
}

// dropDeallocatedStatusDevices removes the status.devices that were allocated
// in the oldClaim and that have been removed in the newClaim.
//
// In other words, it removes stale status entries after deallocation. Doing
// this in the apiserver avoids having to update clients which might be unaware
// of the status feature.
func dropDeallocatedStatusDevices(newClaim, oldClaim *resource.ResourceClaim) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.DRAResourceClaimDeviceStatus) && !isDRAResourceClaimDeviceStatusInUse(oldClaim) {
		return
	}

	deallocatedDevices := sets.New[structured.SharedDeviceID]()

	if oldClaim.Status.Allocation != nil {
		// Get all devices in the oldClaim.
		for _, result := range oldClaim.Status.Allocation.Devices.Results {
			deviceID := structured.MakeDeviceID(result.Driver, result.Pool, result.Device)
			sharedDeviceID := structured.MakeSharedDeviceID(deviceID, result.ShareID)
			deallocatedDevices.Insert(sharedDeviceID)
		}
	}

	// Remove devices from deallocatedDevices that are still in newClaim.
	if newClaim.Status.Allocation != nil {
		for _, result := range newClaim.Status.Allocation.Devices.Results {
			deviceID := structured.MakeDeviceID(result.Driver, result.Pool, result.Device)
			sharedDeviceID := structured.MakeSharedDeviceID(deviceID, result.ShareID)
			deallocatedDevices.Delete(sharedDeviceID)
		}
	}

	// Remove from newClaim.Status.Devices.
	n := 0
	for _, device := range newClaim.Status.Devices {
		deviceID := structured.MakeDeviceID(device.Driver, device.Pool, device.Device)
		var shareID *types.UID
		if device.ShareID != nil {
			shareID = ptr.To(types.UID(*device.ShareID))
		}
		sharedDeviceID := structured.MakeSharedDeviceID(deviceID, shareID)
		if !deallocatedDevices.Has(sharedDeviceID) {
			newClaim.Status.Devices[n] = device
			n++
		}
	}
	newClaim.Status.Devices = newClaim.Status.Devices[:n]

	if len(newClaim.Status.Devices) == 0 {
		newClaim.Status.Devices = nil
	}
}

func draConsumableCapacityFeatureInUse(claim *resource.ResourceClaim) bool {
	if claim == nil {
		return false
	}

	if resourceclaimspec.DRAConsumableCapacityFeatureInUse(&claim.Spec) {
		return true
	}

	if allocation := claim.Status.Allocation; allocation != nil {
		for _, result := range allocation.Devices.Results {
			if result.ShareID != nil || result.ConsumedCapacity != nil {
				return true
			}
		}
	}
	if devices := claim.Status.Devices; devices != nil {
		for _, device := range devices {
			if device.ShareID != nil {
				return true
			}
		}
	}

	return false
}

func draDeviceBindingConditionsInUse(claim *resource.ResourceClaim) bool {
	if claim == nil {
		return false
	}
	if allocation := claim.Status.Allocation; allocation != nil {
		for _, result := range allocation.Devices.Results {
			if result.BindingConditions != nil || result.BindingFailureConditions != nil {
				return true
			}
		}
	}
	return false
}

// dropDisabledDRAResourceClaimConsumableCapacityStatusFields drops any new feature fields
// from the newClaim status if they were not used in the oldClaim.
func dropDisabledDRAResourceClaimConsumableCapacityStatusFields(newClaim, oldClaim *resource.ResourceClaim) {
	if utilfeature.DefaultFeatureGate.Enabled(features.DRAConsumableCapacity) ||
		draConsumableCapacityFeatureInUse(oldClaim) {
		// No need to drop anything.
		return
	}

	if allocation := newClaim.Status.Allocation; allocation != nil {
		for i := range allocation.Devices.Results {
			newClaim.Status.Allocation.Devices.Results[i].ShareID = nil
			newClaim.Status.Allocation.Devices.Results[i].ConsumedCapacity = nil
		}
	}

	if devices := newClaim.Status.Devices; devices != nil {
		for i := range devices {
			newClaim.Status.Devices[i].ShareID = nil
		}
	}
}

// dropDeviceBindingConditionsFields drops any new feature fields
// from the newClaim status if they were not used in the oldClaim.
func dropDeviceBindingConditionsFields(newClaim, oldClaim *resource.ResourceClaim) {
	if utilfeature.DefaultFeatureGate.Enabled(features.DRADeviceBindingConditions) ||
		draDeviceBindingConditionsInUse(oldClaim) {
		// No need to drop anything.
		return
	}

	if allocation := newClaim.Status.Allocation; allocation != nil {
		for i := range allocation.Devices.Results {
			newClaim.Status.Allocation.Devices.Results[i].BindingConditions = nil
			newClaim.Status.Allocation.Devices.Results[i].BindingFailureConditions = nil
		}
	}
}
