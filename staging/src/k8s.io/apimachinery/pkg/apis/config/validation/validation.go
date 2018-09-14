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
	"k8s.io/apimachinery/pkg/apis/config"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// ValidateClientConnectionConfiguration ensures validation of the ClientConnectionConfiguration struct
func ValidateClientConnectionConfiguration(cc *config.ClientConnectionConfiguration, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if cc.Burst < 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("burst"), cc.Burst, "must be non-negative"))
	}
	return allErrs
}
