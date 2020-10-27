/*
Copyright 2020 The Kubernetes Authors.

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
	"fmt"
	"strings"

	apimachineryvalidation "k8s.io/apimachinery/pkg/api/validation"
	"k8s.io/apimachinery/pkg/util/validation"
	utilvalidation "k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/apiserverinternal"
	apivalidation "k8s.io/kubernetes/pkg/apis/core/validation"
)

// ValidateStorageVersion validate the storage version object.
func ValidateStorageVersion(sv *apiserverinternal.StorageVersion) field.ErrorList {
	var allErrs field.ErrorList
	allErrs = append(allErrs, apivalidation.ValidateObjectMeta(&sv.ObjectMeta, false, ValidateStorageVersionName, field.NewPath("metadata"))...)
	allErrs = append(allErrs, validateStorageVersionStatus(sv.Status, field.NewPath("status"))...)
	return allErrs
}

// ValidateStorageVersionName is a ValidateNameFunc for storage version names
func ValidateStorageVersionName(name string, prefix bool) []string {
	var allErrs []string
	idx := strings.LastIndex(name, ".")
	if idx < 0 {
		allErrs = append(allErrs, "name must be in the form of <group>.<resource>")
	} else {
		for _, msg := range utilvalidation.IsDNS1123Subdomain(name[:idx]) {
			allErrs = append(allErrs, "the group segment "+msg)
		}
		for _, msg := range utilvalidation.IsDNS1035Label(name[idx+1:]) {
			allErrs = append(allErrs, "the resource segment "+msg)
		}
	}
	return allErrs
}

// ValidateStorageVersionUpdate tests if an update to a StorageVersion is valid.
func ValidateStorageVersionUpdate(sv, oldSV *apiserverinternal.StorageVersion) field.ErrorList {
	// no error since StorageVersionSpec is an empty spec
	return field.ErrorList{}
}

// ValidateStorageVersionStatusUpdate tests if an update to a StorageVersionStatus is valid.
func ValidateStorageVersionStatusUpdate(sv, oldSV *apiserverinternal.StorageVersion) field.ErrorList {
	var allErrs field.ErrorList
	allErrs = append(allErrs, validateStorageVersionStatus(sv.Status, field.NewPath("status"))...)
	return allErrs
}

func validateStorageVersionStatus(ss apiserverinternal.StorageVersionStatus, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	for i, ssv := range ss.StorageVersions {
		allErrs = append(allErrs, validateServerStorageVersion(ssv, fldPath.Child("storageVersions").Index(i))...)
	}
	if err := validateCommonVersion(ss, fldPath); err != nil {
		allErrs = append(allErrs, err)
	}
	allErrs = append(allErrs, validateStorageVersionCondition(ss.Conditions, fldPath)...)
	return allErrs
}

func validateServerStorageVersion(ssv apiserverinternal.ServerStorageVersion, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for _, msg := range apimachineryvalidation.NameIsDNSSubdomain(ssv.APIServerID, false) {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("apiServerID"), ssv.APIServerID, msg))
	}
	if errs := utilvalidation.IsDNS1035Label(ssv.EncodingVersion); len(errs) > 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("encodingVersion"), ssv.EncodingVersion, strings.Join(errs, ",")))
	}

	found := false
	for i, dv := range ssv.DecodableVersions {
		if errs := utilvalidation.IsDNS1035Label(dv); len(errs) > 0 {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("decodableVersions").Index(i), dv, strings.Join(errs, ",")))
		}
		if dv == ssv.EncodingVersion {
			found = true
		}
	}
	if !found {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("decodableVersions"), ssv.DecodableVersions, fmt.Sprintf("decodableVersions must include encodingVersion %s", ssv.EncodingVersion)))
	}
	return allErrs
}

func commonVersion(ssv []apiserverinternal.ServerStorageVersion) *string {
	if len(ssv) == 0 {
		return nil
	}
	commonVersion := ssv[0].EncodingVersion
	for _, v := range ssv[1:] {
		if v.EncodingVersion != commonVersion {
			return nil
		}
	}
	return &commonVersion
}

func validateCommonVersion(svs apiserverinternal.StorageVersionStatus, fldPath *field.Path) *field.Error {
	actualCommonVersion := commonVersion(svs.StorageVersions)
	if actualCommonVersion == nil && svs.CommonEncodingVersion == nil {
		return nil
	}
	if actualCommonVersion == nil && svs.CommonEncodingVersion != nil {
		return field.Invalid(fldPath.Child("commonEncodingVersion"), *svs.CommonEncodingVersion, "should be nil if servers do not agree on the same encoding version, or if there is no server reporting the supported versions yet")
	}
	if actualCommonVersion != nil && svs.CommonEncodingVersion == nil {
		return field.Invalid(fldPath.Child("commonEncodingVersion"), svs.CommonEncodingVersion, fmt.Sprintf("the common encoding version is %s", *actualCommonVersion))
	}
	if actualCommonVersion != nil && svs.CommonEncodingVersion != nil && *actualCommonVersion != *svs.CommonEncodingVersion {
		return field.Invalid(fldPath.Child("commonEncodingVersion"), *svs.CommonEncodingVersion, fmt.Sprintf("the actual common encoding version is %s", *actualCommonVersion))
	}
	return nil
}

func validateStorageVersionCondition(conditions []apiserverinternal.StorageVersionCondition, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	// We do not verify that the condition type or the condition status is
	// a predefined one because we might add more type or status later.
	seenType := make(map[apiserverinternal.StorageVersionConditionType]int)
	for i, condition := range conditions {
		if ii, ok := seenType[condition.Type]; ok {
			allErrs = append(allErrs, field.Invalid(fldPath.Index(i).Child("type"), string(condition.Type),
				fmt.Sprintf("the type of the condition is not unique, it also appears in conditions[%d]", ii)))
		}
		seenType[condition.Type] = i
		for _, msg := range validation.IsQualifiedName(string(condition.Type)) {
			allErrs = append(allErrs, field.Invalid(fldPath.Index(i).Child("type"), string(condition.Type), msg))
		}
		for _, msg := range validation.IsQualifiedName(string(condition.Status)) {
			allErrs = append(allErrs, field.Invalid(fldPath.Index(i).Child("status"), string(condition.Type), msg))
		}
		if condition.Reason == "" {
			allErrs = append(allErrs, field.Required(fldPath.Index(i).Child("reason"), "reason cannot be empty"))
		}
		if condition.Message == "" {
			allErrs = append(allErrs, field.Required(fldPath.Index(i).Child("message"), "message cannot be empty"))
		}
	}
	return allErrs
}
