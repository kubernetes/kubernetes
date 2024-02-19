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
	"k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// ValidateListOptions returns all validation errors found while validating the ListOptions.
func ValidateListOptions(options *internalversion.ListOptions) field.ErrorList {
	allErrs := field.ErrorList{}
	if match := options.ResourceVersionMatch; len(match) > 0 {
		if options.Watch {
			allErrs = append(allErrs, field.Forbidden(field.NewPath("resourceVersionMatch"), "resourceVersionMatch is forbidden for watch"))
		}
		if len(options.ResourceVersion) == 0 {
			allErrs = append(allErrs, field.Forbidden(field.NewPath("resourceVersionMatch"), "resourceVersionMatch is forbidden unless resourceVersion is provided"))
		}
		if len(options.Continue) > 0 {
			allErrs = append(allErrs, field.Forbidden(field.NewPath("resourceVersionMatch"), "resourceVersionMatch is forbidden when continue is provided"))
		}
		if match != metav1.ResourceVersionMatchExact && match != metav1.ResourceVersionMatchNotOlderThan {
			allErrs = append(allErrs, field.NotSupported(field.NewPath("resourceVersionMatch"), match, []string{string(metav1.ResourceVersionMatchExact), string(metav1.ResourceVersionMatchNotOlderThan), ""}))
		}
		if match == metav1.ResourceVersionMatchExact && options.ResourceVersion == "0" {
			allErrs = append(allErrs, field.Forbidden(field.NewPath("resourceVersionMatch"), "resourceVersionMatch \"exact\" is forbidden for resourceVersion \"0\""))
		}
	}
	return allErrs
}
