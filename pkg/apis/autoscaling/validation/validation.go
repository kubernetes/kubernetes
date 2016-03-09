/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	apivalidation "k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"k8s.io/kubernetes/pkg/util/validation/field"
)

func ValidateScale(scale *autoscaling.Scale) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMeta(&scale.ObjectMeta, true, apivalidation.NameIsDNSSubdomain, field.NewPath("metadata"))...)

	if scale.Spec.Replicas < 0 {
		allErrs = append(allErrs, field.Invalid(field.NewPath("spec", "replicas"), scale.Spec.Replicas, "must be greater than or equal to 0"))
	}

	return allErrs
}
