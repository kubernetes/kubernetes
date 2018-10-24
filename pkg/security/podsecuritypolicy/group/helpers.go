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

package group

import (
	"fmt"

	policy "k8s.io/api/policy/v1beta1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	psputil "k8s.io/kubernetes/pkg/security/podsecuritypolicy/util"
)

func ValidateGroupsInRanges(fldPath *field.Path, ranges []policy.IDRange, groups []int64) field.ErrorList {
	allErrs := field.ErrorList{}

	for _, group := range groups {
		if !isGroupInRanges(group, ranges) {
			detail := fmt.Sprintf("group %d must be in the ranges: %v", group, ranges)
			allErrs = append(allErrs, field.Invalid(fldPath, groups, detail))
		}
	}
	return allErrs
}

func isGroupInRanges(group int64, ranges []policy.IDRange) bool {
	for _, rng := range ranges {
		if psputil.GroupFallsInRange(group, rng) {
			return true
		}
	}
	return false
}
