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

// validateAllowedSchedulingPolicies rejects any compiled PodGroupTemplate whose
// scheduling policy is outside the controller's allow-list.
func (b *Builder) validateAllowedSchedulingPolicies(fldPath *field.Path, workload *schedulingv1alpha3.Workload) field.ErrorList {
	var allErrs field.ErrorList
	policyPath := fldPath.Child("policy")
	for i := range workload.Spec.PodGroupTemplates {
		policy := workload.Spec.PodGroupTemplates[i].SchedulingPolicy
		if policy.Basic != nil && !slices.Contains(b.opts.AllowedPolicies, BasicPolicy) {
			allErrs = append(allErrs, field.Forbidden(
				policyPath.Child("basic"),
				"basic scheduling policy is not supported by this controller"))
		}
		if policy.Gang != nil && !slices.Contains(b.opts.AllowedPolicies, GangPolicy) {
			allErrs = append(allErrs, field.Forbidden(
				policyPath.Child("gang"),
				"gang scheduling policy is not supported by this controller"))
		}
	}
	return allErrs
}

// validateAllowedDisruptionModes rejects any compiled PodGroupTemplate whose
// disruption mode is outside the controller's allow-list.
func (b *Builder) validateAllowedDisruptionModes(fldPath *field.Path, workload *schedulingv1alpha3.Workload) field.ErrorList {
	var allErrs field.ErrorList
	dmPath := fldPath.Child("disruptionMode")
	for i := range workload.Spec.PodGroupTemplates {
		dm := workload.Spec.PodGroupTemplates[i].DisruptionMode
		if dm == nil {
			continue
		}
		if dm.Single != nil && !slices.Contains(b.opts.AllowedDisruptionModes, SingleMode) {
			allErrs = append(allErrs, field.Forbidden(
				dmPath.Child("single"),
				"the disruptionMode `single` is not supported by this controller"))
		}
		if dm.All != nil && !slices.Contains(b.opts.AllowedDisruptionModes, AllMode) {
			allErrs = append(allErrs, field.Forbidden(
				dmPath.Child("all"),
				"the disruptionMode `all` is not supported by this controller"))
		}
	}
	return allErrs
}

// validateDisruptionModeCompatibleWithSchedulingPolicy enforces the cross-field
// rule declarative validation cannot express: a Basic PodGroup is scheduled
// pod-by-pod, so all-or-nothing disruption is meaningless for it.
func (b *Builder) validateDisruptionModeCompatibleWithSchedulingPolicy(fldPath *field.Path, workload *schedulingv1alpha3.Workload) field.ErrorList {
	var allErrs field.ErrorList
	for i := range workload.Spec.PodGroupTemplates {
		tmpl := workload.Spec.PodGroupTemplates[i]
		if tmpl.SchedulingPolicy.Basic != nil && tmpl.DisruptionMode != nil && tmpl.DisruptionMode.All != nil {
			allErrs = append(allErrs, field.Invalid(
				fldPath.Child("disruptionMode", "all"), "",
				"the disruptionMode `all` is not supported with the Basic scheduling policy"))
		}
	}
	return allErrs
}
