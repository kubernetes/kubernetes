/*
Copyright 2017 The Kubernetes Authors.

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
	"k8s.io/kubernetes/pkg/apis/core/validation"
	internalapi "k8s.io/kubernetes/plugin/pkg/admission/priority/apis/priority"
)

// ValidateConfiguration validates the configuration.
func ValidateConfiguration(config *internalapi.Configuration) error {
	if config != nil && len(config.Namespaces) > 0 {
		allErrs := field.ErrorList{}
		fldPath := field.NewPath("priority")
		for _, name := range config.Namespaces {
			for _, msg := range validation.ValidateNamespaceName(name, false) {
				allErrs = append(allErrs, field.Invalid(fldPath.Child("namespace"), name, msg))
			}
		}
		if len(allErrs) > 0 {
			return fmt.Errorf("invalid config: %v", allErrs)
		}
	}
	return nil
}
