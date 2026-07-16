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
	"context"
	"slices"

	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	"k8s.io/apimachinery/pkg/api/operation"
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

// ValidateSchedulingPolicy validates that the set policy is allow-listed by
// the controller and enforces the structural DV rules declared on the API
// type, so out-of-tree controllers get the same validation the kube-apiserver
// applies to built-in types. A nil policy is valid.
func ValidateSchedulingPolicy(
	ctx context.Context,
	policy *schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy,
	fldPath *field.Path,
	allowed ...SchedulingPolicyOption,
) field.ErrorList {
	return validateSchedulingPolicy(ctx, operation.Operation{Type: operation.Create}, policy, nil, fldPath, allowed)
}

// ValidateSchedulingPolicyUpdate enforces the declarative update rules on
// the API type (e.g. gang cannot be set or unset after creation).
// Set-or-unset rules for the policy field as a whole are declared on the
// embedding API's field and remain the controller API's responsibility.
// nil policy is valid here.
func ValidateSchedulingPolicyUpdate(
	ctx context.Context,
	policy, oldPolicy *schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy,
	fldPath *field.Path,
	allowed ...SchedulingPolicyOption,
) field.ErrorList {
	return validateSchedulingPolicy(ctx, operation.Operation{Type: operation.Update}, policy, oldPolicy, fldPath, allowed)
}

func validateSchedulingPolicy(
	ctx context.Context,
	op operation.Operation,
	policy, oldPolicy *schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy,
	fldPath *field.Path,
	allowed []SchedulingPolicyOption,
) field.ErrorList {
	if policy == nil {
		return nil
	}

	var allErrs field.ErrorList

	if policy.Basic != nil && !slices.Contains(allowed, BasicPolicy) {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("basic"), "basic scheduling policy is not supported by this controller"))
	}
	if policy.Gang != nil && !slices.Contains(allowed, GangPolicy) {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("gang"), "gang scheduling policy is not supported by this controller"))
	}

	allErrs = append(allErrs, schedulingv1alpha3.Validate_WorkloadPodGroupSchedulingPolicy(ctx, op, fldPath, policy, oldPolicy)...)

	return allErrs
}

// ValidateDisruptionMode checks that the set mode is in the controller's
// allow-list and enforces the structural declarative validation rules
// declared on the API type (exactly one mode set), so out-of-tree
// controllers get the same validation the kube-apiserver applies to
// built-in types. A nil mode is valid.
func ValidateDisruptionMode(
	ctx context.Context,
	mode *schedulingv1alpha3.WorkloadPodGroupDisruptionMode,
	fldPath *field.Path,
	allowed ...DisruptionModeOption,
) field.ErrorList {
	return validateDisruptionMode(ctx, operation.Operation{Type: operation.Create}, mode, nil, fldPath, allowed)
}

// ValidateDisruptionModeUpdate enforces the declarative update rules on
// the API type. Set-or-unset rules for the mode field as a whole are
// declared on the embedding API's field and remain the controller
// API's responsibility. A nil mode is valid here.
func ValidateDisruptionModeUpdate(
	ctx context.Context,
	mode, oldMode *schedulingv1alpha3.WorkloadPodGroupDisruptionMode,
	fldPath *field.Path,
	allowed ...DisruptionModeOption,
) field.ErrorList {
	return validateDisruptionMode(ctx, operation.Operation{Type: operation.Update}, mode, oldMode, fldPath, allowed)
}

func validateDisruptionMode(
	ctx context.Context,
	op operation.Operation,
	mode, oldMode *schedulingv1alpha3.WorkloadPodGroupDisruptionMode,
	fldPath *field.Path,
	allowed []DisruptionModeOption,
) field.ErrorList {
	if mode == nil {
		return nil
	}

	var allErrs field.ErrorList

	if mode.Single != nil && !slices.Contains(allowed, SingleMode) {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("single"), "single disruption mode is not supported by this controller"))
	}
	if mode.All != nil && !slices.Contains(allowed, AllMode) {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("all"), "all disruption mode is not supported by this controller"))
	}

	allErrs = append(allErrs, schedulingv1alpha3.Validate_WorkloadPodGroupDisruptionMode(ctx, op, fldPath, mode, oldMode)...)

	return allErrs
}
