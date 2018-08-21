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
	"fmt"

	unversionedvalidation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	internalapi "k8s.io/kubernetes/plugin/pkg/admission/noderestriction/apis/noderestriction"
)

// ValidateConfiguration validates the configuration.
func ValidateConfiguration(config *internalapi.Configuration) error {
	allErrs := field.ErrorList{}
	fldPath := field.NewPath("noderestriction")
	for i, key := range config.LabelWhitelist {
		allErrs = append(allErrs, unversionedvalidation.ValidateLabelName(key, fldPath.Index(i).Child("labelWhitelist"))...)
	}
	for i, key := range config.TaintWhitelist {
		allErrs = append(allErrs, unversionedvalidation.ValidateLabelName(key, fldPath.Index(i).Child("taintWhitelist"))...)
	}
	if len(allErrs) > 0 {
		return fmt.Errorf("invalid config: %v", allErrs)
	}
	return nil
}
