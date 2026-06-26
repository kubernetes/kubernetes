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

package workloadbuilder

import (
	"slices"

	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// SchedulingPolicyOption enumerates the scheduling policies a controller opts
// into. The allow-list rejects any policy not listed, so building-block
// additions stay denied until a controller extends its list.
type SchedulingPolicyOption int

const (
	// BasicPolicy allows the Basic scheduling policy.
	BasicPolicy SchedulingPolicyOption = iota
	// GangPolicy allows the Gang scheduling policy.
	GangPolicy
)

// DisruptionModeOption enumerates the disruption modes a controller opts into.
type DisruptionModeOption int

const (
	// SingleMode allows the Single disruption mode.
	SingleMode DisruptionModeOption = iota
	// AllMode allows the All disruption mode.
	AllMode
)

// ValidateSchedulingPolicy checks that exactly one policy is set and that it is
// allow-listed. A nil policy is valid; the controller's defaulting decides the
// effective policy.
func ValidateSchedulingPolicy(
	policy *schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy,
	fldPath *field.Path,
	allowed ...SchedulingPolicyOption,
) field.ErrorList {
	if policy == nil {
		return nil
	}

	var allErrs field.ErrorList
	count := 0

	if policy.Basic != nil {
		count++
		if !slices.Contains(allowed, BasicPolicy) {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("basic"), "basic scheduling policy is not supported by this controller"))
		}
	}
	if policy.Gang != nil {
		count++
		if !slices.Contains(allowed, GangPolicy) {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("gang"), "gang scheduling policy is not supported by this controller"))
		}
		if policy.Gang.MinCount != nil && *policy.Gang.MinCount < 1 {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("gang", "minCount"), *policy.Gang.MinCount, "must be at least 1"))
		}
	}

	switch {
	case count == 0:
		allErrs = append(allErrs, field.Required(fldPath, "exactly one scheduling policy must be set"))
	case count > 1:
		allErrs = append(allErrs, field.Invalid(fldPath, "", "exactly one scheduling policy must be set"))
	}

	return allErrs
}

// ValidateDisruptionMode checks that exactly one disruption mode is set and that
// the chosen mode is in the controller's allow-list. A nil mode is valid.
func ValidateDisruptionMode(
	mode *schedulingv1alpha3.WorkloadPodGroupDisruptionMode,
	fldPath *field.Path,
	allowed ...DisruptionModeOption,
) field.ErrorList {
	if mode == nil {
		return nil
	}

	var allErrs field.ErrorList
	count := 0

	if mode.Single != nil {
		count++
		if !slices.Contains(allowed, SingleMode) {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("single"), "single disruption mode is not supported by this controller"))
		}
	}
	if mode.All != nil {
		count++
		if !slices.Contains(allowed, AllMode) {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("all"), "all disruption mode is not supported by this controller"))
		}
	}

	switch {
	case count == 0:
		allErrs = append(allErrs, field.Required(fldPath, "exactly one disruption mode must be set"))
	case count > 1:
		allErrs = append(allErrs, field.Invalid(fldPath, "", "exactly one disruption mode must be set"))
	}

	return allErrs
}
