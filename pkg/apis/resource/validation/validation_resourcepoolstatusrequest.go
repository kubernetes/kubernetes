/*
Copyright 2026 The Kubernetes Authors.

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
	metav1validation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	corevalidation "k8s.io/kubernetes/pkg/apis/core/validation"
	"k8s.io/kubernetes/pkg/apis/resource"
)

// ValidateResourcePoolStatusRequest validates a ResourcePoolStatusRequest.
func ValidateResourcePoolStatusRequest(request *resource.ResourcePoolStatusRequest) field.ErrorList {
	allErrs := corevalidation.ValidateObjectMeta(&request.ObjectMeta, false, apimachineryvalidation.NameIsDNSSubdomain, field.NewPath("metadata"))
	allErrs = append(allErrs, validateResourcePoolStatusRequestSpec(&request.Spec, field.NewPath("spec"))...)
	return allErrs
}

// ValidateResourcePoolStatusRequestUpdate tests if an update to ResourcePoolStatusRequest is valid.
func ValidateResourcePoolStatusRequestUpdate(request, oldRequest *resource.ResourcePoolStatusRequest) field.ErrorList {
	allErrs := corevalidation.ValidateObjectMetaUpdate(&request.ObjectMeta, &oldRequest.ObjectMeta, field.NewPath("metadata"))
	// The spec is immutable once created.
	allErrs = append(allErrs, apimachineryvalidation.ValidateImmutableField(request.Spec, oldRequest.Spec, field.NewPath("spec"))...)
	return allErrs
}

// ValidateResourcePoolStatusRequestStatusUpdate tests if a status update to ResourcePoolStatusRequest is valid.
func ValidateResourcePoolStatusRequestStatusUpdate(request, oldRequest *resource.ResourcePoolStatusRequest) field.ErrorList {
	allErrs := corevalidation.ValidateObjectMetaUpdate(&request.ObjectMeta, &oldRequest.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, validateResourcePoolStatusRequestStatusUpdate(&request.Status, &oldRequest.Status, field.NewPath("status"))...)
	return allErrs
}

func validateResourcePoolStatusRequestSpec(spec *resource.ResourcePoolStatusRequestSpec, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList

	// Driver is required
	if spec.Driver == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("driver"), "driver name is required"))
	} else {
		allErrs = append(allErrs, validateDriverName(spec.Driver, fldPath.Child("driver"))...)
	}

	// PoolName is optional, but if provided, must be valid
	if spec.PoolName != "" {
		allErrs = append(allErrs, validatePoolName(spec.PoolName, fldPath.Child("poolName"))...)
	}

	// Limit validation
	if spec.Limit != nil {
		if *spec.Limit < 1 {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("limit"), *spec.Limit, "must be at least 1"))
		}
		if *spec.Limit > resource.ResourcePoolStatusRequestLimitMax {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("limit"), *spec.Limit, "must not exceed 1000"))
		}
	}

	return allErrs
}

func validateResourcePoolStatusRequestStatusUpdate(status, oldStatus *resource.ResourcePoolStatusRequestStatus, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList

	// Once observationTime is set, status becomes immutable (request is complete)
	if oldStatus.ObservationTime != nil {
		allErrs = append(allErrs, apimachineryvalidation.ValidateImmutableField(status, oldStatus, fldPath)...)
		return allErrs
	}

	// Validate pools if present
	for i, pool := range status.Pools {
		allErrs = append(allErrs, validatePoolStatus(pool, fldPath.Child("pools").Index(i))...)
	}

	// Validate conditions
	allErrs = append(allErrs, metav1validation.ValidateConditions(status.Conditions, fldPath.Child("conditions"))...)

	// TotalMatchingPools must be non-negative
	if status.TotalMatchingPools < 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("totalMatchingPools"), status.TotalMatchingPools, "must be non-negative"))
	}

	return allErrs
}

func validatePoolStatus(pool resource.PoolStatus, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList

	// Driver is required
	if pool.Driver == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("driver"), ""))
	} else {
		allErrs = append(allErrs, validateDriverName(pool.Driver, fldPath.Child("driver"))...)
	}

	// PoolName is required
	if pool.PoolName == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("poolName"), ""))
	} else {
		allErrs = append(allErrs, validatePoolName(pool.PoolName, fldPath.Child("poolName"))...)
	}

	// NodeName is optional, but if provided, must be valid
	if pool.NodeName != "" {
		allErrs = append(allErrs, validateNodeName(pool.NodeName, fldPath.Child("nodeName"))...)
	}

	// Device counts must be non-negative
	if pool.TotalDevices < 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("totalDevices"), pool.TotalDevices, "must be non-negative"))
	}
	if pool.AllocatedDevices < 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("allocatedDevices"), pool.AllocatedDevices, "must be non-negative"))
	}
	if pool.AvailableDevices < 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("availableDevices"), pool.AvailableDevices, "must be non-negative"))
	}
	if pool.UnavailableDevices < 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("unavailableDevices"), pool.UnavailableDevices, "must be non-negative"))
	}

	// SliceCount must be positive
	if pool.SliceCount < 1 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("sliceCount"), pool.SliceCount, "must be at least 1"))
	}

	// Generation must be non-negative
	if pool.Generation < 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("generation"), pool.Generation, "must be non-negative"))
	}

	return allErrs
}
