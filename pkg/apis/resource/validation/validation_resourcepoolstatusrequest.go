/*
Copyright The Kubernetes Authors.

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
	allErrs = append(allErrs, apimachineryvalidation.ValidateImmutableField(request.Spec, oldRequest.Spec, field.NewPath("spec")).WithOrigin("immutable").MarkCoveredByDeclarative()...)
	return allErrs
}

// ValidateResourcePoolStatusRequestStatusUpdate tests if a status update to ResourcePoolStatusRequest is valid.
func ValidateResourcePoolStatusRequestStatusUpdate(request, oldRequest *resource.ResourcePoolStatusRequest) field.ErrorList {
	allErrs := corevalidation.ValidateObjectMetaUpdate(&request.ObjectMeta, &oldRequest.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, validateResourcePoolStatusRequestStatusUpdate(request.Status, oldRequest.Status, field.NewPath("status"))...)
	return allErrs
}

func validateResourcePoolStatusRequestSpec(spec *resource.ResourcePoolStatusRequestSpec, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList

	// Driver is required - covered by +k8s:required and +k8s:format=k8s-long-name DV tags
	if spec.Driver == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("driver"), "driver name is required").MarkCoveredByDeclarative())
	} else {
		allErrs = append(allErrs, validateDriverName(spec.Driver, fldPath.Child("driver"), corevalidation.FormatCovered, corevalidation.SizeCovered)...)
	}

	// PoolName is optional, but if provided, must be non-empty and valid
	// Covered by +k8s:format=k8s-resource-pool-name DV tag
	if spec.PoolName != nil {
		if *spec.PoolName == "" {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("poolName"), *spec.PoolName, "must not be empty when specified").MarkCoveredByDeclarative())
		} else {
			allErrs = append(allErrs, validatePoolName(*spec.PoolName, fldPath.Child("poolName")).MarkCoveredByDeclarative()...)
		}
	}

	// Limit validation - covered by +k8s:minimum=1 and +k8s:maximum=1000 DV tags
	if spec.Limit != nil {
		if *spec.Limit < 1 {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("limit"), *spec.Limit, "must be at least 1").MarkCoveredByDeclarative())
		}
		if *spec.Limit > resource.ResourcePoolStatusRequestLimitMax {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("limit"), *spec.Limit, "must not exceed 1000").MarkCoveredByDeclarative())
		}
	}

	return allErrs
}

func validateResourcePoolStatusRequestStatusUpdate(status, oldStatus *resource.ResourcePoolStatusRequestStatus, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList

	// Once status is set, it becomes immutable (request is complete)
	if oldStatus != nil {
		allErrs = append(allErrs, apimachineryvalidation.ValidateImmutableField(status, oldStatus, fldPath)...)
		return allErrs
	}

	// If new status is nil, nothing to validate
	if status == nil {
		return allErrs
	}

	// Validate pools if present
	for i, pool := range status.Pools {
		allErrs = append(allErrs, validatePoolStatus(pool, fldPath.Child("pools").Index(i))...)
	}

	// Validate validation errors: max 10 entries, max 256 chars each
	// Covered by +k8s:maxItems=10 and +k8s:eachVal=+k8s:maxLength=256 DV tags
	if len(status.ValidationErrors) > 10 {
		allErrs = append(allErrs, field.TooMany(fldPath.Child("validationErrors"), len(status.ValidationErrors), 10).MarkCoveredByDeclarative())
	}
	for i, msg := range status.ValidationErrors {
		if len(msg) > 256 {
			allErrs = append(allErrs, field.TooLong(fldPath.Child("validationErrors").Index(i), msg, 256).MarkCoveredByDeclarative())
		}
	}

	// Validate conditions
	allErrs = append(allErrs, metav1validation.ValidateConditions(status.Conditions, fldPath.Child("conditions"))...)

	// TotalMatchingPools must be non-negative if specified
	// Covered by +k8s:minimum=0 DV tag
	if status.TotalMatchingPools != nil && *status.TotalMatchingPools < 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("totalMatchingPools"), *status.TotalMatchingPools, "must be non-negative").MarkCoveredByDeclarative())
	}

	return allErrs
}

func validatePoolStatus(pool resource.PoolStatus, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList

	// Driver is required — covered by +k8s:required and +k8s:format DV tags
	if pool.Driver == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("driver"), "").MarkCoveredByDeclarative())
	} else {
		allErrs = append(allErrs, validateDriverName(pool.Driver, fldPath.Child("driver"), corevalidation.FormatCovered, corevalidation.SizeCovered)...)
	}

	// PoolName is required — covered by +k8s:required and +k8s:format DV tags
	if pool.PoolName == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("poolName"), "").MarkCoveredByDeclarative())
	} else {
		allErrs = append(allErrs, validatePoolName(pool.PoolName, fldPath.Child("poolName")).MarkCoveredByDeclarative()...)
	}

	// NodeName is optional, but if provided, must be valid
	// Covered by +k8s:format=k8s-long-name DV tag
	if pool.NodeName != nil {
		if *pool.NodeName == "" {
			allErrs = append(allErrs, field.Required(fldPath.Child("nodeName"), "nodeName must not be empty when specified").MarkCoveredByDeclarative())
		} else {
			allErrs = append(allErrs, validateNodeName(*pool.NodeName, fldPath.Child("nodeName"))...)
		}
	}

	// Required pointer fields must not be nil — covered by +k8s:required DV tags
	// Minimum values covered by +k8s:minimum=0 DV tags
	if pool.TotalDevices == nil {
		allErrs = append(allErrs, field.Required(fldPath.Child("totalDevices"), "").MarkCoveredByDeclarative())
	} else if *pool.TotalDevices < 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("totalDevices"), *pool.TotalDevices, "must be non-negative").MarkCoveredByDeclarative())
	}
	if pool.AllocatedDevices == nil {
		allErrs = append(allErrs, field.Required(fldPath.Child("allocatedDevices"), "").MarkCoveredByDeclarative())
	} else if *pool.AllocatedDevices < 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("allocatedDevices"), *pool.AllocatedDevices, "must be non-negative").MarkCoveredByDeclarative())
	}
	if pool.AvailableDevices == nil {
		allErrs = append(allErrs, field.Required(fldPath.Child("availableDevices"), "").MarkCoveredByDeclarative())
	} else if *pool.AvailableDevices < 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("availableDevices"), *pool.AvailableDevices, "must be non-negative").MarkCoveredByDeclarative())
	}
	if pool.UnavailableDevices == nil {
		allErrs = append(allErrs, field.Required(fldPath.Child("unavailableDevices"), "").MarkCoveredByDeclarative())
	} else if *pool.UnavailableDevices < 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("unavailableDevices"), *pool.UnavailableDevices, "must be non-negative").MarkCoveredByDeclarative())
	}

	// SliceCount must be positive — covered by +k8s:minimum=1 DV tag
	if pool.SliceCount < 1 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("sliceCount"), pool.SliceCount, "must be at least 1").MarkCoveredByDeclarative())
	}

	// Generation is required and must be non-negative — covered by +k8s:required and +k8s:minimum=0 DV tags
	if pool.Generation == nil {
		allErrs = append(allErrs, field.Required(fldPath.Child("generation"), "").MarkCoveredByDeclarative())
	} else if *pool.Generation < 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("generation"), *pool.Generation, "must be non-negative").MarkCoveredByDeclarative())
	}

	return allErrs
}
