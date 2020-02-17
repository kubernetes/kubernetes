/*
Copyright 2019 The Kubernetes Authors.

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

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/apis/apiserverinternal"
)

// ValidateStorageVersion validate the storage version object.
func ValidateStorageVersion(sv *apiserverinternal.StorageVersion) field.ErrorList {
	var allErrs field.ErrorList
	ssvPath := field.NewPath("status.serverStorageVersions")
	for i, ssv := range sv.Status.ServerStorageVersions {
		if err := validateServerStorageVersion(ssv, ssvPath.Index(i)); err != nil {
			allErrs = append(allErrs, err)
		}
	}
	if err := validateAllEncodingVersionsEqual(sv.Status, field.NewPath("status")); err != nil {
		allErrs = append(allErrs, err)
	}
	return allErrs
}

func validateServerStorageVersion(ssv apiserverinternal.ServerStorageVersion, fldPath *field.Path) *field.Error {
	if ssv.APIServerID == "" {
		return field.Invalid(fldPath.Child("apiServerID"), ssv.APIServerID, "apiServerID cannot be empty")
	}
	if ssv.EncodingVersion == "" {
		return field.Invalid(fldPath.Child("encodingVersion"), ssv.EncodingVersion, "encodingVersion cannot be empty")
	}
	found := false
	for _, dv := range ssv.DecodableVersions {
		if dv == ssv.EncodingVersion {
			found = true
			break
		}
	}
	if !found {
		return field.Invalid(fldPath.Child("decodableVersions"), ssv.DecodableVersions, fmt.Sprintf("decodableVersions must include encodingVersion %s", ssv.EncodingVersion))
	}
	return nil
}

func agreedVersion(ssv []apiserverinternal.ServerStorageVersion) *string {
	if len(ssv) == 0 {
		return nil
	}
	agreedVersion := ssv[0].EncodingVersion
	for _, v := range ssv[1:] {
		if v.EncodingVersion != agreedVersion {
			return nil
		}
	}
	return &agreedVersion
}

func validateAllEncodingVersionsEqual(svs apiserverinternal.StorageVersionStatus, fldPath *field.Path) *field.Error {
	actualAgreedVersion := agreedVersion(svs.ServerStorageVersions)
	if actualAgreedVersion == nil && svs.AgreedEncodingVersion == nil {
		return nil
	}
	if actualAgreedVersion == nil && svs.AgreedEncodingVersion != nil {
		return field.Invalid(fldPath.Child("agreedEncodingVersion"), *svs.AgreedEncodingVersion, "should be nil if servers do not agree on the same encoding version, or if there is no server reporting the supported versions yet")
	}
	if actualAgreedVersion != nil && svs.AgreedEncodingVersion == nil {
		return field.Invalid(fldPath.Child("agreedEncodingVersion"), svs.AgreedEncodingVersion, fmt.Sprintf("the agreed version is %s", *actualAgreedVersion))
	}
	if actualAgreedVersion != nil && svs.AgreedEncodingVersion != nil && *actualAgreedVersion != *svs.AgreedEncodingVersion {
		return field.Invalid(fldPath.Child("agreedEncodingVersion"), *svs.AgreedEncodingVersion, fmt.Sprintf("the actual agreed version is %s", *actualAgreedVersion))
	}
	return nil
}
