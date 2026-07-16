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

	"k8s.io/apimachinery/pkg/util/validation/field"
)

// validateAllowedSchedulingPolicies rejects any resolved config whose
// scheduling policy is outside the controller's allow-list.
func (b *Builder) validateAllowedSchedulingPolicies(item *WorkloadItem, resolvedConfig *SchedulingConfig, rootPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList

	if resolvedConfig == nil || resolvedConfig.Policy == nil {
		return nil
	}

	path := appendPathElements(rootPath, item.Input.Policy.PathElements)

	policy := resolvedConfig.Policy
	if policy.Basic != nil && !slices.Contains(b.opts.AllowedPolicies, BasicPolicy) {
		allErrs = append(allErrs, field.Forbidden(
			path,
			"basic scheduling policy is not supported by this controller"))
	}
	if policy.Gang != nil && !slices.Contains(b.opts.AllowedPolicies, GangPolicy) {
		allErrs = append(allErrs, field.Forbidden(
			path,
			"gang scheduling policy is not supported by this controller"))
	}
	return allErrs
}

// validateAllowedDisruptionModes rejects any resolved config whose
// disruption mode is outside the controller's allow-list.
func (b *Builder) validateAllowedDisruptionModes(item *WorkloadItem, resolvedConfig *SchedulingConfig, rootPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList

	if resolvedConfig == nil || resolvedConfig.DisruptionMode == nil {
		return nil
	}

	path := appendPathElements(rootPath, item.Input.DisruptionMode.PathElements)

	dm := resolvedConfig.DisruptionMode
	if dm.Single != nil && !slices.Contains(b.opts.AllowedDisruptionModes, SingleMode) {
		allErrs = append(allErrs, field.Forbidden(
			path,
			"the disruptionMode `single` is not supported by this controller"))
	}
	if dm.All != nil && !slices.Contains(b.opts.AllowedDisruptionModes, AllMode) {
		allErrs = append(allErrs, field.Forbidden(
			path,
			"the disruptionMode `all` is not supported by this controller"))
	}
	return allErrs
}

// validateDisruptionModeCompatibleWithSchedulingPolicy enforces the cross-field
// rule declarative validation cannot express: a Basic PodGroup is scheduled
// pod-by-pod, so all-or-nothing disruption is meaningless for it.
func (b *Builder) validateDisruptionModeCompatibleWithSchedulingPolicy(item *WorkloadItem, resolvedConfig *SchedulingConfig, rootPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList

	if resolvedConfig == nil {
		return nil
	}

	path := appendPathElements(rootPath, item.Input.DisruptionMode.PathElements)

	if resolvedConfig.Policy != nil && resolvedConfig.Policy.Basic != nil &&
		resolvedConfig.DisruptionMode != nil && resolvedConfig.DisruptionMode.All != nil {
		allErrs = append(allErrs, field.Invalid(
			path, "",
			"the disruptionMode `all` is not supported with the Basic scheduling policy"))
	}

	return allErrs
}
