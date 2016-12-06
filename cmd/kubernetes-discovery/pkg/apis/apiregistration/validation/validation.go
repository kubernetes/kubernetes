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

	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/api/validation/path"
	utilvalidation "k8s.io/kubernetes/pkg/util/validation"
	"k8s.io/kubernetes/pkg/util/validation/field"

	discoveryapi "k8s.io/kubernetes/cmd/kubernetes-discovery/pkg/apis/apiregistration"
)

func ValidateAPIService(apiServer *discoveryapi.APIService) field.ErrorList {
	requiredName := apiServer.Spec.Version + "." + apiServer.Spec.Group

	allErrs := validation.ValidateObjectMeta(&apiServer.ObjectMeta, false,
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
	if len(apiServer.Spec.Group) == 0 && apiServer.Spec.Version != "v1" {
		allErrs = append(allErrs, field.Required(field.NewPath("spec", "group"), "only v1 may have an empty group and it better be legacy kube"))
	}
	if len(apiServer.Spec.Group) > 0 {
		for _, errString := range utilvalidation.IsDNS1123Subdomain(apiServer.Spec.Group) {
			allErrs = append(allErrs, field.Invalid(field.NewPath("spec", "group"), apiServer.Spec.Group, errString))
		}
	}

	for _, errString := range utilvalidation.IsDNS1035Label(apiServer.Spec.Version) {
		allErrs = append(allErrs, field.Invalid(field.NewPath("spec", "version"), apiServer.Spec.Version, errString))
	}

	if apiServer.Spec.Priority <= 0 || apiServer.Spec.Priority > 1000 {
		allErrs = append(allErrs, field.Invalid(field.NewPath("spec", "priority"), apiServer.Spec.Priority, "priority must be positive and less than 1000"))

	}

	if len(apiServer.Spec.Service.Namespace) == 0 {
		allErrs = append(allErrs, field.Required(field.NewPath("spec", "service", "namespace"), ""))
	}
	if len(apiServer.Spec.Service.Name) == 0 {
		allErrs = append(allErrs, field.Required(field.NewPath("spec", "service", "name"), ""))
	}

	if apiServer.Spec.InsecureSkipTLSVerify && len(apiServer.Spec.CABundle) > 0 {
		allErrs = append(allErrs, field.Invalid(field.NewPath("spec", "insecureSkipTLSVerify"), apiServer.Spec.InsecureSkipTLSVerify, "may not be true if caBundle is present"))
	}

	return allErrs
}

func ValidateAPIServiceUpdate(newAPIService *discoveryapi.APIService, oldAPIService *discoveryapi.APIService) field.ErrorList {
	allErrs := validation.ValidateObjectMetaUpdate(&newAPIService.ObjectMeta, &oldAPIService.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateAPIService(newAPIService)...)

	return allErrs
}
