/*
Copyright 2016 The Kubernetes Authors.

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

	"k8s.io/apimachinery/pkg/api/validation"
	"k8s.io/apimachinery/pkg/api/validation/path"
	utilvalidation "k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"

	discoveryapi "k8s.io/kube-aggregator/pkg/apis/apiregistration"
)

func ValidateAPIService(apiService *discoveryapi.APIService) field.ErrorList {
	requiredName := apiService.Spec.Version + "." + apiService.Spec.Group

	allErrs := validation.ValidateObjectMeta(&apiService.ObjectMeta, false,
		func(name string, prefix bool) []string {
			if minimalFailures := path.IsValidPathSegmentName(name); len(minimalFailures) > 0 {
				return minimalFailures
			}
			// the name *must* be version.group
			if name != requiredName {
				return []string{fmt.Sprintf("must be `spec.version+\".\"+spec.group`: %q", requiredName)}
			}

			return []string{}
		},
		field.NewPath("metadata"))

	// in this case we allow empty group
	if len(apiService.Spec.Group) == 0 && apiService.Spec.Version != "v1" {
		allErrs = append(allErrs, field.Required(field.NewPath("spec", "group"), "only v1 may have an empty group and it better be legacy kube"))
	}
	if len(apiService.Spec.Group) > 0 {
		for _, errString := range utilvalidation.IsDNS1123Subdomain(apiService.Spec.Group) {
			allErrs = append(allErrs, field.Invalid(field.NewPath("spec", "group"), apiService.Spec.Group, errString))
		}
	}

	for _, errString := range utilvalidation.IsDNS1035Label(apiService.Spec.Version) {
		allErrs = append(allErrs, field.Invalid(field.NewPath("spec", "version"), apiService.Spec.Version, errString))
	}

	if apiService.Spec.Priority <= 0 || apiService.Spec.Priority > 1000 {
		allErrs = append(allErrs, field.Invalid(field.NewPath("spec", "priority"), apiService.Spec.Priority, "priority must be positive and less than 1000"))

	}

	if apiService.Spec.Service == nil {
		if len(apiService.Spec.CABundle) != 0 {
			allErrs = append(allErrs, field.Invalid(field.NewPath("spec", "caBundle"), fmt.Sprintf("%d bytes", len(apiService.Spec.CABundle)), "local APIServices may not have a caBundle"))
		}
		if apiService.Spec.InsecureSkipTLSVerify {
			allErrs = append(allErrs, field.Invalid(field.NewPath("spec", "insecureSkipTLSVerify"), apiService.Spec.InsecureSkipTLSVerify, "local APIServices may not have insecureSkipTLSVerify"))
		}
		return allErrs
	}

	if len(apiService.Spec.Service.Namespace) == 0 {
		allErrs = append(allErrs, field.Required(field.NewPath("spec", "service", "namespace"), ""))
	}
	if len(apiService.Spec.Service.Name) == 0 {
		allErrs = append(allErrs, field.Required(field.NewPath("spec", "service", "name"), ""))
	}
	if apiService.Spec.InsecureSkipTLSVerify && len(apiService.Spec.CABundle) > 0 {
		allErrs = append(allErrs, field.Invalid(field.NewPath("spec", "insecureSkipTLSVerify"), apiService.Spec.InsecureSkipTLSVerify, "may not be true if caBundle is present"))
	}

	return allErrs
}

func ValidateAPIServiceUpdate(newAPIService *discoveryapi.APIService, oldAPIService *discoveryapi.APIService) field.ErrorList {
	allErrs := validation.ValidateObjectMetaUpdate(&newAPIService.ObjectMeta, &oldAPIService.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateAPIService(newAPIService)...)

	return allErrs
}
