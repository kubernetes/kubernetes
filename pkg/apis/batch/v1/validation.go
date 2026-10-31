/*
Copyright The Kubernetes Authors.

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

package v1

import (
	"context"

	batchv1 "k8s.io/api/batch/v1"
	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	batchvalidation "k8s.io/kubernetes/pkg/apis/batch/validation"
)

// ValidateCustom_PodFailurePolicy_Rules checks generated condition reason syntax
// and cross-rule collisions, which cannot be expressed with built-in formats.
func ValidateCustom_PodFailurePolicy_Rules(_ context.Context, _ operation.Operation, fldPath *field.Path, rules, _ []batchv1.PodFailurePolicyRule) field.ErrorList {
	var allErrs field.ErrorList
	seen := sets.New[string]()
	for i, rule := range rules {
		if rule.Name != nil {
			allErrs = append(allErrs, batchvalidation.ValidatePodFailurePolicyRuleName(*rule.Name, i, len(rules), seen, fldPath.Index(i).Child("name"))...)
		}
	}
	return allErrs
}
