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

package validation

import (
	"k8s.io/apimachinery/pkg/api/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/coordination"
)

// ValidateLease validates a Lease.
func ValidateLease(lease *coordination.Lease) field.ErrorList {
	allErrs := validation.ValidateObjectMeta(&lease.ObjectMeta, true, validation.NameIsDNSSubdomain, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateLeaseSpec(&lease.Spec, field.NewPath("spec"))...)
	return allErrs
}

// ValidateLeaseUpdate validates an update of Lease object.
func ValidateLeaseUpdate(lease, oldLease *coordination.Lease) field.ErrorList {
	allErrs := validation.ValidateObjectMetaUpdate(&lease.ObjectMeta, &oldLease.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateLeaseSpec(&lease.Spec, field.NewPath("spec"))...)
	return allErrs
}

// ValidateLeaseSpec validates spec of Lease.
func ValidateLeaseSpec(spec *coordination.LeaseSpec, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if spec.LeaseDurationSeconds != nil && *spec.LeaseDurationSeconds <= 0 {
		fld := fldPath.Child("leaseDurationSeconds")
		allErrs = append(allErrs, field.Invalid(fld, spec.LeaseDurationSeconds, "must be greater than 0"))
	}
	if spec.LeaseTransitions != nil && *spec.LeaseTransitions < 0 {
		fld := fldPath.Child("leaseTransitions")
		allErrs = append(allErrs, field.Invalid(fld, spec.LeaseTransitions, "must be greater than or equal to 0"))
	}
	return allErrs
}

// ValidateIdentityLease validates a IdentityLease.
func ValidateIdentityLease(lease *coordination.IdentityLease) field.ErrorList {
	allErrs := validation.ValidateObjectMeta(&lease.ObjectMeta, true, validation.NameIsDNSSubdomain, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateIdentityLeaseSpec(&lease.Spec, field.NewPath("spec"))...)
	return allErrs
}

// ValidateIdentityLeaseUpdate validates an update of IdentityLease object.
func ValidateIdentityLeaseUpdate(lease, oldIdentityLease *coordination.IdentityLease) field.ErrorList {
	allErrs := validation.ValidateObjectMetaUpdate(&lease.ObjectMeta, &oldIdentityLease.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateIdentityLeaseSpec(&lease.Spec, field.NewPath("spec"))...)
	return allErrs
}

// ValidateIdentityLeaseSpec validates spec of IdentityLease.
func ValidateIdentityLeaseSpec(spec *coordination.IdentityLeaseSpec, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if spec.LeaseDurationSeconds != nil && *spec.LeaseDurationSeconds <= 0 {
		fld := fldPath.Child("leaseDurationSeconds")
		allErrs = append(allErrs, field.Invalid(fld, spec.LeaseDurationSeconds, "must be greater than 0"))
	}
	return allErrs
}
