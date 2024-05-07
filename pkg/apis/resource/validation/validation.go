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

package validation

import (
	apimachineryvalidation "k8s.io/apimachinery/pkg/api/validation"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	corevalidation "k8s.io/kubernetes/pkg/apis/core/validation"
	"k8s.io/kubernetes/pkg/apis/resource"
)

// validateResourceDriverName reuses the validation of a CSI driver because
// the allowed values are exactly the same.
var validateResourceDriverName = corevalidation.ValidateCSIDriverName

// ValidateClaim validates a ResourceClaim.
func ValidateClaim(resourceClaim *resource.ResourceClaim) field.ErrorList {
	allErrs := corevalidation.ValidateObjectMeta(&resourceClaim.ObjectMeta, true, corevalidation.ValidateResourceClaimName, field.NewPath("metadata"))
	allErrs = append(allErrs, validateResourceClaimSpec(&resourceClaim.Spec, field.NewPath("spec"))...)
	return allErrs
}

func validateResourceClaimSpec(spec *resource.ResourceClaimSpec, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for _, msg := range corevalidation.ValidateClassName(spec.ResourceClassName, false) {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("resourceClassName"), spec.ResourceClassName, msg))
	}
	allErrs = append(allErrs, validateResourceClaimParameters(spec.ParametersRef, fldPath.Child("parametersRef"))...)
	if !supportedAllocationModes.Has(string(spec.AllocationMode)) {
		allErrs = append(allErrs, field.NotSupported(fldPath.Child("allocationMode"), spec.AllocationMode, supportedAllocationModes.List()))
	}
	return allErrs
}

var supportedAllocationModes = sets.NewString(string(resource.AllocationModeImmediate), string(resource.AllocationModeWaitForFirstConsumer))

// It would have been nice to use Go generics to reuse the same validation
// function for Kind and Name in both types, but generics cannot be used to
// access common fields in structs.

func validateResourceClaimParameters(ref *resource.ResourceClaimParametersReference, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	if ref != nil {
		if ref.Kind == "" {
			allErrs = append(allErrs, field.Required(fldPath.Child("kind"), ""))
		}
		if ref.Name == "" {
			allErrs = append(allErrs, field.Required(fldPath.Child("name"), ""))
		}
	}
	return allErrs
}

func validateClassParameters(ref *resource.ResourceClassParametersReference, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	if ref != nil {
		if ref.Kind == "" {
			allErrs = append(allErrs, field.Required(fldPath.Child("kind"), ""))
		}
		if ref.Name == "" {
			allErrs = append(allErrs, field.Required(fldPath.Child("name"), ""))
		}
		if ref.Namespace != "" {
			for _, msg := range apimachineryvalidation.ValidateNamespaceName(ref.Namespace, false) {
				allErrs = append(allErrs, field.Invalid(fldPath.Child("namespace"), ref.Namespace, msg))
			}
		}
	}
	return allErrs
}

// ValidateClass validates a ResourceClass.
func ValidateClass(resourceClass *resource.ResourceClass) field.ErrorList {
	allErrs := corevalidation.ValidateObjectMeta(&resourceClass.ObjectMeta, false, corevalidation.ValidateClassName, field.NewPath("metadata"))
	allErrs = append(allErrs, validateResourceDriverName(resourceClass.DriverName, field.NewPath("driverName"))...)
	allErrs = append(allErrs, validateClassParameters(resourceClass.ParametersRef, field.NewPath("parametersRef"))...)
	if resourceClass.SuitableNodes != nil {
		allErrs = append(allErrs, corevalidation.ValidateNodeSelector(resourceClass.SuitableNodes, field.NewPath("suitableNodes"))...)
	}

	return allErrs
}

// ValidateClassUpdate tests if an update to ResourceClass is valid.
func ValidateClassUpdate(resourceClass, oldClass *resource.ResourceClass) field.ErrorList {
	allErrs := corevalidation.ValidateObjectMetaUpdate(&resourceClass.ObjectMeta, &oldClass.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateClass(resourceClass)...)
	return allErrs
}

// ValidateClaimUpdate tests if an update to ResourceClaim is valid.
func ValidateClaimUpdate(resourceClaim, oldClaim *resource.ResourceClaim) field.ErrorList {
	allErrs := corevalidation.ValidateObjectMetaUpdate(&resourceClaim.ObjectMeta, &oldClaim.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, apimachineryvalidation.ValidateImmutableField(resourceClaim.Spec, oldClaim.Spec, field.NewPath("spec"))...)
	allErrs = append(allErrs, ValidateClaim(resourceClaim)...)
	return allErrs
}

// ValidateClaimStatusUpdate tests if an update to the status of a ResourceClaim is valid.
func ValidateClaimStatusUpdate(resourceClaim, oldClaim *resource.ResourceClaim) field.ErrorList {
	allErrs := corevalidation.ValidateObjectMetaUpdate(&resourceClaim.ObjectMeta, &oldClaim.ObjectMeta, field.NewPath("metadata"))
	fldPath := field.NewPath("status")
	// The name might not be set yet.
	if resourceClaim.Status.DriverName != "" {
		allErrs = append(allErrs, validateResourceDriverName(resourceClaim.Status.DriverName, fldPath.Child("driverName"))...)
	} else if resourceClaim.Status.Allocation != nil {
		allErrs = append(allErrs, field.Required(fldPath.Child("driverName"), "must be specified when `allocation` is set"))
	}

	allErrs = append(allErrs, validateAllocationResult(resourceClaim.Status.Allocation, fldPath.Child("allocation"))...)
	allErrs = append(allErrs, validateResourceClaimConsumers(resourceClaim.Status.ReservedFor, resource.ResourceClaimReservedForMaxSize, fldPath.Child("reservedFor"))...)

	// Now check for invariants that must be valid for a ResourceClaim.
	if len(resourceClaim.Status.ReservedFor) > 0 {
		if resourceClaim.Status.Allocation == nil {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("reservedFor"), "may not be specified when `allocated` is not set"))
		} else {
			if !resourceClaim.Status.Allocation.Shareable && len(resourceClaim.Status.ReservedFor) > 1 {
				allErrs = append(allErrs, field.Forbidden(fldPath.Child("reservedFor"), "may not be reserved more than once"))
			}
			// Items may be removed from ReservedFor while the claim is meant to be deallocated,
			// but not added.
			if resourceClaim.DeletionTimestamp != nil || resourceClaim.Status.DeallocationRequested {
				oldSet := sets.New(oldClaim.Status.ReservedFor...)
				newSet := sets.New(resourceClaim.Status.ReservedFor...)
				newItems := newSet.Difference(oldSet)
				if len(newItems) > 0 {
					allErrs = append(allErrs, field.Forbidden(fldPath.Child("reservedFor"), "new entries may not be added while `deallocationRequested` or `deletionTimestamp` are set"))
				}
			}
		}
	}

	// Updates to a populated resourceClaim.Status.Allocation are not allowed
	if oldClaim.Status.Allocation != nil && resourceClaim.Status.Allocation != nil {
		allErrs = append(allErrs, apimachineryvalidation.ValidateImmutableField(resourceClaim.Status.Allocation, oldClaim.Status.Allocation, fldPath.Child("allocation"))...)
	}

	if !oldClaim.Status.DeallocationRequested &&
		resourceClaim.Status.DeallocationRequested &&
		len(resourceClaim.Status.ReservedFor) > 0 {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("deallocationRequested"), "deallocation cannot be requested while `reservedFor` is set"))
	}

	if resourceClaim.Status.Allocation == nil &&
		resourceClaim.Status.DeallocationRequested {
		// Either one or the other field was modified incorrectly.
		// For the sake of simplicity this only reports the invalid
		// end result.
		allErrs = append(allErrs, field.Forbidden(fldPath, "`allocation` must be set when `deallocationRequested` is set"))
	}

	// Once deallocation has been requested, that request cannot be removed
	// anymore because the deallocation may already have started. The field
	// can only get reset by the driver together with removing the
	// allocation.
	if oldClaim.Status.DeallocationRequested &&
		!resourceClaim.Status.DeallocationRequested &&
		resourceClaim.Status.Allocation != nil {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("deallocationRequested"), "may not be cleared when `allocation` is set"))
	}

	return allErrs
}

func validateAllocationResult(allocation *resource.AllocationResult, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	if allocation != nil {
		if len(allocation.ResourceHandles) > 0 {
			allErrs = append(allErrs, validateResourceHandles(allocation.ResourceHandles, resource.AllocationResultResourceHandlesMaxSize, fldPath.Child("resourceHandles"))...)
		}
		if allocation.AvailableOnNodes != nil {
			allErrs = append(allErrs, corevalidation.ValidateNodeSelector(allocation.AvailableOnNodes, fldPath.Child("availableOnNodes"))...)
		}
	}
	return allErrs
}

func validateResourceHandles(resourceHandles []resource.ResourceHandle, maxSize int, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	for i, resourceHandle := range resourceHandles {
		idxPath := fldPath.Index(i)
		allErrs = append(allErrs, validateResourceDriverName(resourceHandle.DriverName, idxPath.Child("driverName"))...)
		if len(resourceHandle.Data) > resource.ResourceHandleDataMaxSize {
			allErrs = append(allErrs, field.TooLongMaxLength(idxPath.Child("data"), len(resourceHandle.Data), resource.ResourceHandleDataMaxSize))
		}
	}
	if len(resourceHandles) > maxSize {
		// Dumping the entire field into the error message is likely to be too long,
		// in particular when it is already beyond the maximum size. Instead this
		// just shows the number of entries.
		allErrs = append(allErrs, field.TooLongMaxLength(fldPath, len(resourceHandles), maxSize))
	}
	return allErrs
}

func validateResourceClaimUserReference(ref resource.ResourceClaimConsumerReference, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	if ref.Resource == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("resource"), ""))
	}
	if ref.Name == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("name"), ""))
	}
	if ref.UID == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("uid"), ""))
	}
	return allErrs
}

// validateSliceIsASet ensures that a slice contains no duplicates and does not exceed a certain maximum size.
func validateSliceIsASet[T comparable](slice []T, maxSize int, validateItem func(item T, fldPath *field.Path) field.ErrorList, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	allItems := sets.New[T]()
	for i, item := range slice {
		idxPath := fldPath.Index(i)
		if allItems.Has(item) {
			allErrs = append(allErrs, field.Duplicate(idxPath, item))
		} else {
			allErrs = append(allErrs, validateItem(item, idxPath)...)
			allItems.Insert(item)
		}
	}
	if len(slice) > maxSize {
		// Dumping the entire field into the error message is likely to be too long,
		// in particular when it is already beyond the maximum size. Instead this
		// just shows the number of entries.
		allErrs = append(allErrs, field.TooLongMaxLength(fldPath, len(slice), maxSize))
	}
	return allErrs
}

// validateResourceClaimConsumers ensures that the slice contains no duplicate UIDs and does not exceed a certain maximum size.
func validateResourceClaimConsumers(consumers []resource.ResourceClaimConsumerReference, maxSize int, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	allUIDs := sets.New[types.UID]()
	for i, consumer := range consumers {
		idxPath := fldPath.Index(i)
		if allUIDs.Has(consumer.UID) {
			allErrs = append(allErrs, field.Duplicate(idxPath.Child("uid"), consumer.UID))
		} else {
			allErrs = append(allErrs, validateResourceClaimUserReference(consumer, idxPath)...)
			allUIDs.Insert(consumer.UID)
		}
	}
	if len(consumers) > maxSize {
		// Dumping the entire field into the error message is likely to be too long,
		// in particular when it is already beyond the maximum size. Instead this
		// just shows the number of entries.
		allErrs = append(allErrs, field.TooLongMaxLength(fldPath, len(consumers), maxSize))
	}
	return allErrs
}

// ValidatePodSchedulingContext validates a PodSchedulingContext.
func ValidatePodSchedulingContexts(schedulingCtx *resource.PodSchedulingContext) field.ErrorList {
	allErrs := corevalidation.ValidateObjectMeta(&schedulingCtx.ObjectMeta, true, corevalidation.ValidatePodName, field.NewPath("metadata"))
	allErrs = append(allErrs, validatePodSchedulingSpec(&schedulingCtx.Spec, field.NewPath("spec"))...)
	return allErrs
}

func validatePodSchedulingSpec(spec *resource.PodSchedulingContextSpec, fldPath *field.Path) field.ErrorList {
	allErrs := validateSliceIsASet(spec.PotentialNodes, resource.PodSchedulingNodeListMaxSize, validateNodeName, fldPath.Child("potentialNodes"))
	return allErrs
}

// ValidatePodSchedulingContextUpdate tests if an update to PodSchedulingContext is valid.
func ValidatePodSchedulingContextUpdate(schedulingCtx, oldSchedulingCtx *resource.PodSchedulingContext) field.ErrorList {
	allErrs := corevalidation.ValidateObjectMetaUpdate(&schedulingCtx.ObjectMeta, &oldSchedulingCtx.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidatePodSchedulingContexts(schedulingCtx)...)
	return allErrs
}

// ValidatePodSchedulingContextStatusUpdate tests if an update to the status of a PodSchedulingContext is valid.
func ValidatePodSchedulingContextStatusUpdate(schedulingCtx, oldSchedulingCtx *resource.PodSchedulingContext) field.ErrorList {
	allErrs := corevalidation.ValidateObjectMetaUpdate(&schedulingCtx.ObjectMeta, &oldSchedulingCtx.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, validatePodSchedulingStatus(&schedulingCtx.Status, field.NewPath("status"))...)
	return allErrs
}

func validatePodSchedulingStatus(status *resource.PodSchedulingContextStatus, fldPath *field.Path) field.ErrorList {
	return validatePodSchedulingClaims(status.ResourceClaims, fldPath.Child("claims"))
}

func validatePodSchedulingClaims(claimStatuses []resource.ResourceClaimSchedulingStatus, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	names := sets.NewString()
	for i, claimStatus := range claimStatuses {
		allErrs = append(allErrs, validatePodSchedulingClaim(claimStatus, fldPath.Index(i))...)
		if names.Has(claimStatus.Name) {
			allErrs = append(allErrs, field.Duplicate(fldPath.Index(i), claimStatus.Name))
		} else {
			names.Insert(claimStatus.Name)
		}
	}
	return allErrs
}

func validatePodSchedulingClaim(status resource.ResourceClaimSchedulingStatus, fldPath *field.Path) field.ErrorList {
	allErrs := validateSliceIsASet(status.UnsuitableNodes, resource.PodSchedulingNodeListMaxSize, validateNodeName, fldPath.Child("unsuitableNodes"))
	return allErrs
}

// ValidateClaimTemplace validates a ResourceClaimTemplate.
func ValidateClaimTemplate(template *resource.ResourceClaimTemplate) field.ErrorList {
	allErrs := corevalidation.ValidateObjectMeta(&template.ObjectMeta, true, corevalidation.ValidateResourceClaimTemplateName, field.NewPath("metadata"))
	allErrs = append(allErrs, validateResourceClaimTemplateSpec(&template.Spec, field.NewPath("spec"))...)
	return allErrs
}

func validateResourceClaimTemplateSpec(spec *resource.ResourceClaimTemplateSpec, fldPath *field.Path) field.ErrorList {
	allErrs := corevalidation.ValidateTemplateObjectMeta(&spec.ObjectMeta, fldPath.Child("metadata"))
	allErrs = append(allErrs, validateResourceClaimSpec(&spec.Spec, fldPath.Child("spec"))...)
	return allErrs
}

// ValidateClaimTemplateUpdate tests if an update to template is valid.
func ValidateClaimTemplateUpdate(template, oldTemplate *resource.ResourceClaimTemplate) field.ErrorList {
	allErrs := corevalidation.ValidateObjectMetaUpdate(&template.ObjectMeta, &oldTemplate.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, apimachineryvalidation.ValidateImmutableField(template.Spec, oldTemplate.Spec, field.NewPath("spec"))...)
	allErrs = append(allErrs, ValidateClaimTemplate(template)...)
	return allErrs
}

func validateNodeName(name string, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	for _, msg := range corevalidation.ValidateNodeName(name, false) {
		allErrs = append(allErrs, field.Invalid(fldPath, name, msg))
	}
	return allErrs
}
