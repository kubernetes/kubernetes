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
	"fmt"
	"slices"

	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// ValidationInput carries the parameters Validate only consults when
// declarative validation is enabled. Its zero value means a create with no
// previous object, which is the common case; a caller running with
// DisableDeclarativeValidation can always pass the zero value.
type ValidationInput struct {
	// OldRoot is the previously persisted WorkloadItem. It is nil for a create
	// and non-nil for an update. Validate infers the operation from it, so the
	// update-time declarative checks run exactly when OldRoot is set.
	//
	// Only OldRoot.Name (to correlate it with the new root) and OldRoot.Input
	// (the previous versioned data) are consulted. DefaultConfig, Callbacks, and
	// Input.*.PathElements are ignored on the old root because the resolved
	// config and error paths always come from the new root. Callers may therefore
	// leave those fields unset on the old root.
	OldRoot *WorkloadItem
}

// Validate runs declarative validation on the input blocks (unless disabled)
// and controller-policy checks that declarative validation cannot express.
// For create operations, pass the zero ValidationInput. For update operations,
// set OldRoot to the previous WorkloadItem. Validate infers create vs update
// from whether OldRoot is nil.
func (b *Builder) Validate(ctx context.Context, rootPath *field.Path, input ValidationInput) field.ErrorList {
	// A Builder created from an existing, persisted Workload has no WorkloadItem
	// tree to check, so calling Validate on it is an invocation error rather than
	// a success: the object already passed apiserver validation and this builder
	// is only meant to materialize PodGroups.
	if b.existingWorkload != nil {
		return field.ErrorList{field.InternalError(rootPath,
			fmt.Errorf("cannot validate a builder constructed from an existing workload"))}
	}
	if b.root == nil {
		return field.ErrorList{field.Invalid(rootPath, nil, "invalid builder: missing root WorkloadItem")}
	}
	if rootPath == nil {
		rootPath = field.NewPath("")
	}

	// Old and new roots are correlated by name. A root name mismatch means the
	// caller paired unrelated trees, which is an invocation-contract bug rather
	// than a user-facing validation error.
	if input.OldRoot != nil && input.OldRoot.Name != b.root.Name {
		return field.ErrorList{field.InternalError(rootPath,
			fmt.Errorf("old root name %q does not match new root name %q", input.OldRoot.Name, b.root.Name))}
	}

	var allErrs field.ErrorList
	if !b.opts.DisableDeclarativeValidation {
		allErrs = append(allErrs, b.validateDeclarative(ctx, input.OldRoot, rootPath)...)
	}

	resolvedConfig := resolveSchedulingConfig(b.root)
	allErrs = append(allErrs, b.validateAllowedSchedulingPolicies(b.root, resolvedConfig, rootPath)...)
	allErrs = append(allErrs, b.validateAllowedDisruptionModes(b.root, resolvedConfig, rootPath)...)
	allErrs = append(allErrs, b.validateDisruptionModeCompatibleWithSchedulingPolicy(b.root, resolvedConfig, rootPath)...)
	return allErrs
}

// appendPathElements returns rootPath with the given elements appended as
// children. If elements is empty, it returns the root path itself.
func appendPathElements(rootPath *field.Path, elements []string) *field.Path {
	path := rootPath
	for _, el := range elements {
		path = path.Child(el)
	}
	return path
}

func (b *Builder) validateDeclarative(ctx context.Context, oldRoot *WorkloadItem, rootPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList

	// A non-nil old root means this is an update, so declarative validation
	// runs its update-time checks against the previous data.
	op := operation.Operation{Type: operation.Create}
	if oldRoot != nil {
		op = operation.Operation{Type: operation.Update}
	}

	if b.root.Input.Policy.PodGroupData != nil {
		path := appendPathElements(rootPath, b.root.Input.Policy.PathElements)
		var oldData *schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy
		if oldRoot != nil && oldRoot.Input.Policy.PodGroupData != nil {
			oldData = oldRoot.Input.Policy.PodGroupData
		}
		allErrs = append(allErrs, schedulingv1alpha3.Validate_WorkloadPodGroupSchedulingPolicy(ctx, op, path, b.root.Input.Policy.PodGroupData, oldData)...)
	}
	if b.root.Input.Constraints.PodGroupData != nil {
		path := appendPathElements(rootPath, b.root.Input.Constraints.PathElements)
		var oldData *schedulingv1alpha3.WorkloadPodGroupSchedulingConstraints
		if oldRoot != nil && oldRoot.Input.Constraints.PodGroupData != nil {
			oldData = oldRoot.Input.Constraints.PodGroupData
		}
		allErrs = append(allErrs, schedulingv1alpha3.Validate_WorkloadPodGroupSchedulingConstraints(ctx, op, path, b.root.Input.Constraints.PodGroupData, oldData)...)
	}
	if b.root.Input.DisruptionMode.PodGroupData != nil {
		path := appendPathElements(rootPath, b.root.Input.DisruptionMode.PathElements)
		var oldData *schedulingv1alpha3.WorkloadPodGroupDisruptionMode
		if oldRoot != nil && oldRoot.Input.DisruptionMode.PodGroupData != nil {
			oldData = oldRoot.Input.DisruptionMode.PodGroupData
		}
		allErrs = append(allErrs, schedulingv1alpha3.Validate_WorkloadPodGroupDisruptionMode(ctx, op, path, b.root.Input.DisruptionMode.PodGroupData, oldData)...)
	}

	allErrs = append(allErrs, b.validateResourceClaims(ctx, op, oldRoot, rootPath)...)

	return allErrs
}

// validateResourceClaims runs declarative validation on each resource claim
// building block. Resource claims are keyed by name rather than position, so an
// old claim is matched to its new counterpart by name for the update-time
// checks; a new claim with no old match validates as an addition.
func (b *Builder) validateResourceClaims(ctx context.Context, op operation.Operation, oldRoot *WorkloadItem, rootPath *field.Path) field.ErrorList {
	newClaims := b.root.Input.ResourceClaims.PodGroupData
	var oldClaims []schedulingv1alpha3.WorkloadPodGroupResourceClaim
	if oldRoot != nil {
		oldClaims = oldRoot.Input.ResourceClaims.PodGroupData
	}
	if len(newClaims) == 0 && len(oldClaims) == 0 {
		return nil
	}

	path := appendPathElements(rootPath, b.root.Input.ResourceClaims.PathElements)

	oldByName := make(map[string]*schedulingv1alpha3.WorkloadPodGroupResourceClaim, len(oldClaims))
	for i := range oldClaims {
		oldByName[oldClaims[i].Name] = &oldClaims[i]
	}

	var allErrs field.ErrorList
	for idx := range newClaims {
		data := &newClaims[idx]
		oldData := oldByName[data.Name]
		allErrs = append(allErrs, schedulingv1alpha3.Validate_WorkloadPodGroupResourceClaim(ctx, op, path.Index(idx), data, oldData)...)
	}
	return allErrs
}

// validateAllowedSchedulingPolicies rejects a resolved scheduling policy that is
// outside the controller's allow-list, reporting at the policy block's path.
func (b *Builder) validateAllowedSchedulingPolicies(item *WorkloadItem, resolvedConfig *SchedulingConfig, rootPath *field.Path) field.ErrorList {
	if resolvedConfig == nil || resolvedConfig.Policy == nil {
		return nil
	}

	path := appendPathElements(rootPath, item.Input.Policy.PathElements)

	var allErrs field.ErrorList
	policy := resolvedConfig.Policy
	if policy.Basic != nil && !slices.Contains(b.opts.AllowedPolicies, BasicPolicy) {
		allErrs = append(allErrs, field.Forbidden(
			path, "basic scheduling policy is not supported by this controller"))
	}
	if policy.Gang != nil && !slices.Contains(b.opts.AllowedPolicies, GangPolicy) {
		allErrs = append(allErrs, field.Forbidden(
			path, "gang scheduling policy is not supported by this controller"))
	}
	return allErrs
}

// validateAllowedDisruptionModes rejects a resolved disruption mode that is
// outside the controller's allow-list, reporting at the disruptionMode block's path.
func (b *Builder) validateAllowedDisruptionModes(item *WorkloadItem, resolvedConfig *SchedulingConfig, rootPath *field.Path) field.ErrorList {
	if resolvedConfig == nil || resolvedConfig.DisruptionMode == nil {
		return nil
	}

	path := appendPathElements(rootPath, item.Input.DisruptionMode.PathElements)

	var allErrs field.ErrorList
	dm := resolvedConfig.DisruptionMode
	if dm.Single != nil && !slices.Contains(b.opts.AllowedDisruptionModes, SingleMode) {
		allErrs = append(allErrs, field.Forbidden(
			path, "the disruptionMode `single` is not supported by this controller"))
	}
	if dm.All != nil && !slices.Contains(b.opts.AllowedDisruptionModes, AllMode) {
		allErrs = append(allErrs, field.Forbidden(
			path, "the disruptionMode `all` is not supported by this controller"))
	}
	return allErrs
}

// validateDisruptionModeCompatibleWithSchedulingPolicy enforces the cross-field
// rule declarative validation cannot express: a Basic PodGroup is scheduled
// pod-by-pod, so all-or-nothing disruption is meaningless for it.
func (b *Builder) validateDisruptionModeCompatibleWithSchedulingPolicy(item *WorkloadItem, resolvedConfig *SchedulingConfig, rootPath *field.Path) field.ErrorList {
	if resolvedConfig == nil ||
		resolvedConfig.Policy == nil || resolvedConfig.Policy.Basic == nil ||
		resolvedConfig.DisruptionMode == nil || resolvedConfig.DisruptionMode.All == nil {
		return nil
	}

	path := appendPathElements(rootPath, item.Input.DisruptionMode.PathElements)
	return field.ErrorList{field.Invalid(
		path, "", "the disruptionMode `all` is not supported with the Basic scheduling policy")}
}
