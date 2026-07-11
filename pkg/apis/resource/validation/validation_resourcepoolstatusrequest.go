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
	return allErrs
}

// ValidateResourcePoolStatusRequestUpdate tests if an update to ResourcePoolStatusRequest is valid.
func ValidateResourcePoolStatusRequestUpdate(request, oldRequest *resource.ResourcePoolStatusRequest) field.ErrorList {
	allErrs := corevalidation.ValidateObjectMetaUpdate(&request.ObjectMeta, &oldRequest.ObjectMeta, field.NewPath("metadata"))
	return allErrs
}

// ValidateResourcePoolStatusRequestStatusUpdate tests if a status update to ResourcePoolStatusRequest is valid.
func ValidateResourcePoolStatusRequestStatusUpdate(request, oldRequest *resource.ResourcePoolStatusRequest) field.ErrorList {
	allErrs := corevalidation.ValidateObjectMetaUpdate(&request.ObjectMeta, &oldRequest.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, validateResourcePoolStatusRequestStatusUpdate(request.Status, oldRequest.Status, field.NewPath("status"))...)
	return allErrs
}

func validateResourcePoolStatusRequestStatusUpdate(status, oldStatus *resource.ResourcePoolStatusRequestStatus, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList

	// Once status is set, it becomes immutable (request is complete) — not DV covered
	if oldStatus != nil {
		allErrs = append(allErrs, apimachineryvalidation.ValidateImmutableField(status, oldStatus, fldPath)...)
		return allErrs
	}

	// If new status is nil, nothing to validate
	if status == nil {
		return allErrs
	}

	// Validate conditions — not DV covered
	allErrs = append(allErrs, metav1validation.ValidateConditions(status.Conditions, fldPath.Child("conditions"))...)

	return allErrs
}
