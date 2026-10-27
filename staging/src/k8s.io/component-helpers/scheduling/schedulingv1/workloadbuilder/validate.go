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
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
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
	// (the previous versioned data) are consulted. Path, DefaultConfig, Callbacks,
	// and Input.*.PathElements are ignored on the old root because the resolved
	// config and error paths always come from the new root. Callers may therefore
	// leave those fields unset on the old root.
	OldRoot *WorkloadItem
}

// Validate runs declarative validation on the input blocks (unless disabled)
// and controller-policy checks that declarative validation cannot express.
// For create operations, pass the zero ValidationInput. For update operations,
// set OldRoot to the previous WorkloadItem. Validate infers create vs update
// from whether OldRoot is nil. Each item's building-block errors are reported
// relative to that item's Path.
func (b *Builder) Validate(ctx context.Context, input ValidationInput) field.ErrorList {
	// A Builder created from an existing, persisted Workload has no WorkloadItem
	// tree to check, so calling Validate on it is an invocation error rather than
	// a success: the object already passed apiserver validation and this builder
	// is only meant to materialize PodGroups.
	if b.existingWorkload != nil {
		return field.ErrorList{field.InternalError(nil,
			fmt.Errorf("cannot validate a builder constructed from an existing workload"))}
	}
	if b.root == nil {
		return field.ErrorList{field.Invalid(nil, nil, "invalid builder: missing root WorkloadItem")}
	}

	rootPath := b.root.Path

	// Old and new roots are correlated by name. A root name mismatch means the
	// caller paired unrelated trees, which is an invocation-contract bug rather
	// than a user-facing validation error.
	if input.OldRoot != nil && input.OldRoot.Name != b.root.Name {
		return field.ErrorList{field.InternalError(rootPath,
			fmt.Errorf("old root name %q does not match new root name %q", input.OldRoot.Name, b.root.Name))}
	}

	if err := validateWorkloadItemTree(b.root); err != nil {
		return field.ErrorList{field.InternalError(rootPath, err)}
	}

	op := operation.Operation{Type: operation.Create}
	if input.OldRoot != nil {
		op = operation.Operation{Type: operation.Update}
	}

	allErrs := b.validateItem(ctx, op, b.root, input.OldRoot, itemValidationOpts{})
	allErrs = append(allErrs, b.validatePriorityClassName(b.root)...)
	return allErrs
}

// validateWorkloadItemTree does a full traversal of the tree.
func validateWorkloadItemTree(root *WorkloadItem) error {
	seen := make(map[string]struct{})

	var validate func(item *WorkloadItem, depth int) error
	validate = func(item *WorkloadItem, depth int) error {
		if item == nil {
			return fmt.Errorf("workload item cannot be nil")
		}
		if item.Name == "" {
			return fmt.Errorf("workload item name cannot be empty")
		}
		if _, dup := seen[item.Name]; dup {
			return fmt.Errorf("duplicate workload item name %q: template names must be unique across the Workload", item.Name)
		}
		seen[item.Name] = struct{}{}

		if depth > schedulingv1beta1.WorkloadMaxTreeDepth {
			return fmt.Errorf("workload item %q at depth %d exceeds maximum allowed tree depth of %d", item.Name, depth, schedulingv1beta1.WorkloadMaxTreeDepth)
		}

		if err := validateItemInputUnion(item); err != nil {
			return err
		}

		var leafChildren, compositeChildren int
		for _, child := range item.Children {
			if err := validate(child, depth+1); err != nil {
				return err
			}
			if isComposite(child) {
				compositeChildren++
			} else {
				leafChildren++
			}
		}
		if leafChildren > schedulingv1beta1.WorkloadMaxPodGroupTemplates {
			return fmt.Errorf("workload item %q has %d leaf children, exceeding maximum allowed of %d", item.Name, leafChildren, schedulingv1beta1.WorkloadMaxPodGroupTemplates)
		}
		if compositeChildren > schedulingv1beta1.WorkloadMaxPodGroupTemplates {
			return fmt.Errorf("workload item %q has %d composite children, exceeding maximum allowed of %d", item.Name, compositeChildren, schedulingv1beta1.WorkloadMaxPodGroupTemplates)
		}
		return nil
	}

	return validate(root, 1)
}

// validateItemInputUnion enforces the WorkloadInput leaf/composite union. A composite
// node must not set any leaf building block, and a leaf node must not set any
// composite building block.
func validateItemInputUnion(item *WorkloadItem) error {
	in := item.Input

	if isComposite(item) {
		if in.Policy.PodGroupData != nil {
			return fmt.Errorf("composite workload item %q sets leaf building block: policy", item.Name)
		}
		if in.Constraints.PodGroupData != nil {
			return fmt.Errorf("composite workload item %q sets leaf building block: constraints", item.Name)
		}
		if in.DisruptionMode.PodGroupData != nil {
			return fmt.Errorf("composite workload item %q sets leaf building block: disruptionMode", item.Name)
		}
		if len(in.ResourceClaims.PodGroupData) > 0 {
			return fmt.Errorf("composite workload item %q sets leaf building block: resourceClaims", item.Name)
		}
		return nil
	}

	if in.Policy.CompositePodGroupData != nil {
		return fmt.Errorf("leaf workload item %q sets composite building block: policy", item.Name)
	}
	if in.Constraints.CompositePodGroupData != nil {
		return fmt.Errorf("leaf workload item %q sets composite building block: constraints", item.Name)
	}
	if in.DisruptionMode.CompositePodGroupData != nil {
		return fmt.Errorf("leaf workload item %q sets composite building block: disruptionMode", item.Name)
	}

	return nil
}

type itemValidationOpts struct {
	requireGangSchedulingPolicy bool
	requireAllDisruptionMode    bool
}

// validateItem runs declarative and controller-policy validation on a single
// WorkloadItem and recurses into its children. Each node reports its
// building-block errors relative to its own Path, and controllers map the errors
// onto their own API fields.
func (b *Builder) validateItem(ctx context.Context, op operation.Operation, item, oldItem *WorkloadItem, opts itemValidationOpts) field.ErrorList {
	var allErrs field.ErrorList

	if !b.opts.DisableDeclarativeValidation {
		allErrs = append(allErrs, b.validateItemDeclarative(ctx, op, item, oldItem)...)
	}

	resolvedConfig := resolveSchedulingConfig(item)
	allErrs = append(allErrs, b.validateAllowedSchedulingPolicies(item, resolvedConfig)...)
	allErrs = append(allErrs, b.validateAllowedDisruptionModes(item, resolvedConfig)...)
	allErrs = append(allErrs, b.validateDisruptionModeCompatibleWithSchedulingPolicy(item, resolvedConfig)...)
	allErrs = append(allErrs, b.validateParentChildSchedulingPolicy(item, resolvedConfig, opts.requireGangSchedulingPolicy)...)
	allErrs = append(allErrs, b.validateParentChildDisruptionMode(item, resolvedConfig, opts.requireAllDisruptionMode)...)

	// oldItem is the previous counterpart of item (or nil for a create or a node
	// that is new in this update). Children are correlated with their previous
	// counterparts by name within this single level only, so a node that keeps its
	// name but changes its position in the hierarchy is treated as an addition.
	var oldChildrenByName map[string]*WorkloadItem
	if oldItem != nil {
		oldChildrenByName = make(map[string]*WorkloadItem, len(oldItem.Children))
		for _, oldChild := range oldItem.Children {
			oldChildrenByName[oldChild.Name] = oldChild
		}
	}

	if hasGangPolicy(resolvedConfig) {
		opts.requireGangSchedulingPolicy = true
	}
	if hasAllDisruptionMode(resolvedConfig) {
		opts.requireAllDisruptionMode = true
	}
	for _, child := range item.Children {
		allErrs = append(allErrs, b.validateItem(ctx, op, child, oldChildrenByName[child.Name], opts)...)
	}
	return allErrs
}

// validatePriorityClassName ensures all items in the tree resolve to the same
// PriorityClassName, reporting a single error at the root path if any diverge.
// Keep in sync with validateWorkloadPriority in
// k8s.io/kubernetes/pkg/apis/scheduling/validation.
func (b *Builder) validatePriorityClassName(root *WorkloadItem) field.ErrorList {
	rootPriority := resolveSchedulingConfig(root).PriorityClassName

	var hasMismatch func(item *WorkloadItem) bool
	hasMismatch = func(item *WorkloadItem) bool {
		if resolveSchedulingConfig(item).PriorityClassName != rootPriority {
			return true
		}
		return slices.ContainsFunc(item.Children, hasMismatch)
	}

	if slices.ContainsFunc(root.Children, hasMismatch) {
		return field.ErrorList{field.Invalid(
			root.Path, field.OmitValueType{}, "all priorityClassName values must be identical, detected multiple values")}
	}
	return nil
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

// validateItemDeclarative runs declarative validation on a single node's input
// building blocks, dispatching to the leaf or composite block set based on
// whether the node is a composite group.
func (b *Builder) validateItemDeclarative(ctx context.Context, op operation.Operation, item, oldItem *WorkloadItem) field.ErrorList {
	if isComposite(item) {
		return b.validateCompositePodGroupItemDeclarative(ctx, op, item, oldItem)
	}
	return b.validatePodGroupItemDeclarative(ctx, op, item, oldItem)
}

// validatePodGroupItemDeclarative runs declarative validation on a leaf node's
// building blocks.
func (b *Builder) validatePodGroupItemDeclarative(ctx context.Context, op operation.Operation,
	item, oldItem *WorkloadItem) field.ErrorList {
	var allErrs field.ErrorList

	if item.Input.Policy.PodGroupData != nil {
		path := appendPathElements(item.Path, item.Input.Policy.PathElements)
		var oldData *schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy
		if oldItem != nil && oldItem.Input.Policy.PodGroupData != nil {
			oldData = oldItem.Input.Policy.PodGroupData
		}
		allErrs = append(allErrs, schedulingv1alpha3.Validate_WorkloadPodGroupSchedulingPolicy(ctx, op, path, item.Input.Policy.PodGroupData, oldData)...)
	}
	if item.Input.Constraints.PodGroupData != nil {
		path := appendPathElements(item.Path, item.Input.Constraints.PathElements)
		var oldData *schedulingv1alpha3.WorkloadPodGroupSchedulingConstraints
		if oldItem != nil && oldItem.Input.Constraints.PodGroupData != nil {
			oldData = oldItem.Input.Constraints.PodGroupData
		}
		allErrs = append(allErrs, schedulingv1alpha3.Validate_WorkloadPodGroupSchedulingConstraints(ctx, op, path, item.Input.Constraints.PodGroupData, oldData)...)
	}
	if item.Input.DisruptionMode.PodGroupData != nil {
		path := appendPathElements(item.Path, item.Input.DisruptionMode.PathElements)
		var oldData *schedulingv1alpha3.WorkloadPodGroupDisruptionMode
		if oldItem != nil && oldItem.Input.DisruptionMode.PodGroupData != nil {
			oldData = oldItem.Input.DisruptionMode.PodGroupData
		}
		allErrs = append(allErrs, schedulingv1alpha3.Validate_WorkloadPodGroupDisruptionMode(ctx, op, path, item.Input.DisruptionMode.PodGroupData, oldData)...)
	}

	allErrs = append(allErrs, b.validateResourceClaims(ctx, op, item, oldItem)...)

	return allErrs
}

// validateCompositePodGroupItemDeclarative runs declarative validation on a
// composite node's building blocks.
func (b *Builder) validateCompositePodGroupItemDeclarative(ctx context.Context, op operation.Operation, item, oldItem *WorkloadItem) field.ErrorList {
	var allErrs field.ErrorList

	if item.Input.Policy.CompositePodGroupData != nil {
		path := appendPathElements(item.Path, item.Input.Policy.PathElements)
		var oldData *schedulingv1alpha3.WorkloadCompositePodGroupSchedulingPolicy
		if oldItem != nil && oldItem.Input.Policy.CompositePodGroupData != nil {
			oldData = oldItem.Input.Policy.CompositePodGroupData
		}
		allErrs = append(allErrs, schedulingv1alpha3.Validate_WorkloadCompositePodGroupSchedulingPolicy(ctx, op, path, item.Input.Policy.CompositePodGroupData, oldData)...)
	}
	if item.Input.Constraints.CompositePodGroupData != nil {
		path := appendPathElements(item.Path, item.Input.Constraints.PathElements)
		var oldData *schedulingv1alpha3.WorkloadCompositePodGroupSchedulingConstraints
		if oldItem != nil && oldItem.Input.Constraints.CompositePodGroupData != nil {
			oldData = oldItem.Input.Constraints.CompositePodGroupData
		}
		allErrs = append(allErrs, schedulingv1alpha3.Validate_WorkloadCompositePodGroupSchedulingConstraints(ctx, op, path, item.Input.Constraints.CompositePodGroupData, oldData)...)
	}
	if item.Input.DisruptionMode.CompositePodGroupData != nil {
		path := appendPathElements(item.Path, item.Input.DisruptionMode.PathElements)
		var oldData *schedulingv1alpha3.WorkloadCompositePodGroupDisruptionMode
		if oldItem != nil && oldItem.Input.DisruptionMode.CompositePodGroupData != nil {
			oldData = oldItem.Input.DisruptionMode.CompositePodGroupData
		}
		allErrs = append(allErrs, schedulingv1alpha3.Validate_WorkloadCompositePodGroupDisruptionMode(ctx, op, path, item.Input.DisruptionMode.CompositePodGroupData, oldData)...)
	}

	return allErrs
}

// validateResourceClaims runs declarative validation on each resource claim
// building block. Resource claims are keyed by name rather than position, so an
// old claim is matched to its new counterpart by name for the update-time
// checks; a new claim with no old match validates as an addition.
func (b *Builder) validateResourceClaims(ctx context.Context, op operation.Operation, item, oldItem *WorkloadItem) field.ErrorList {
	newClaims := item.Input.ResourceClaims.PodGroupData
	var oldClaims []schedulingv1alpha3.WorkloadPodGroupResourceClaim
	if oldItem != nil {
		oldClaims = oldItem.Input.ResourceClaims.PodGroupData
	}
	if len(newClaims) == 0 && len(oldClaims) == 0 {
		return nil
	}

	path := appendPathElements(item.Path, item.Input.ResourceClaims.PathElements)

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
// An unset policy compiles to Basic, so it is checked against BasicPolicy.
func (b *Builder) validateAllowedSchedulingPolicies(item *WorkloadItem, resolvedConfig *SchedulingConfig) field.ErrorList {
	path := appendPathElements(item.Path, item.Input.Policy.PathElements)

	var allErrs field.ErrorList
	if hasBasicPolicy(resolvedConfig) && !slices.Contains(b.opts.AllowedPolicies, BasicPolicy) {
		allErrs = append(allErrs, field.Forbidden(
			path, "basic scheduling policy is not supported by this controller"))
	}
	if hasGangPolicy(resolvedConfig) && !slices.Contains(b.opts.AllowedPolicies, GangPolicy) {
		allErrs = append(allErrs, field.Forbidden(
			path, "gang scheduling policy is not supported by this controller"))
	}
	return allErrs
}

// validateAllowedDisruptionModes rejects a resolved disruption mode that is
// outside the controller's allow-list, reporting at the disruptionMode block's path.
// An unset mode defaults to Single, so it is checked against SingleMode.
func (b *Builder) validateAllowedDisruptionModes(item *WorkloadItem, resolvedConfig *SchedulingConfig) field.ErrorList {
	path := appendPathElements(item.Path, item.Input.DisruptionMode.PathElements)

	var allErrs field.ErrorList
	if hasSingleDisruptionMode(resolvedConfig) && !slices.Contains(b.opts.AllowedDisruptionModes, SingleMode) {
		allErrs = append(allErrs, field.Forbidden(
			path, "the disruptionMode `single` is not supported by this controller"))
	}
	if hasAllDisruptionMode(resolvedConfig) && !slices.Contains(b.opts.AllowedDisruptionModes, AllMode) {
		allErrs = append(allErrs, field.Forbidden(
			path, "the disruptionMode `all` is not supported by this controller"))
	}
	return allErrs
}

// validateDisruptionModeCompatibleWithSchedulingPolicy enforces the cross-field
// rule declarative validation cannot express: a Basic group (including an unset
// policy, which compiles to Basic) is scheduled independently, so
// all-or-nothing disruption is meaningless for it.
func (b *Builder) validateDisruptionModeCompatibleWithSchedulingPolicy(item *WorkloadItem, resolvedConfig *SchedulingConfig) field.ErrorList {
	if !hasBasicPolicy(resolvedConfig) || !hasAllDisruptionMode(resolvedConfig) {
		return nil
	}

	path := appendPathElements(item.Path, item.Input.DisruptionMode.PathElements)
	return field.ErrorList{field.Invalid(
		path, "", "the disruptionMode `all` is not supported with the Basic scheduling policy")}
}

// validateParentChildSchedulingPolicy rejects a child group that resolves to a
// non-gang scheduling policy (including an unset policy, which compiles to
// Basic) when an ancestor composite group uses Gang scheduling.
func (b *Builder) validateParentChildSchedulingPolicy(item *WorkloadItem, resolvedConfig *SchedulingConfig, requireGang bool) field.ErrorList {
	if !requireGang || hasGangPolicy(resolvedConfig) {
		return nil
	}

	path := appendPathElements(item.Path, item.Input.Policy.PathElements)
	return field.ErrorList{field.Invalid(
		path, field.OmitValueType{}, "must have `gang` scheduling policy when parent group has `gang` scheduling policy")}
}

// hasGangPolicy reports whether the config resolves to the Gang scheduling
// policy.
func hasGangPolicy(cfg *SchedulingConfig) bool {
	return cfg != nil && cfg.Policy != nil && cfg.Policy.Gang != nil
}

// hasBasicPolicy reports whether the config resolves to the Basic scheduling
// policy, either explicitly or by defaulting when no policy is set.
func hasBasicPolicy(cfg *SchedulingConfig) bool {
	if cfg == nil || cfg.Policy == nil || *cfg.Policy == (SchedulingPolicy{}) {
		return true
	}
	return cfg.Policy.Basic != nil
}

// validateParentChildDisruptionMode rejects a child group that resolves to a
// non-All disruption mode (including an unset mode, which defaults to Single)
// when an ancestor composite group uses the All disruption mode.
func (b *Builder) validateParentChildDisruptionMode(item *WorkloadItem, resolvedConfig *SchedulingConfig, requireAll bool) field.ErrorList {
	if !requireAll || hasAllDisruptionMode(resolvedConfig) {
		return nil
	}

	path := appendPathElements(item.Path, item.Input.DisruptionMode.PathElements)
	return field.ErrorList{field.Invalid(
		path, field.OmitValueType{}, "must have `all` disruption mode when parent group has `all` disruption mode")}
}

// hasAllDisruptionMode reports whether the config resolves to the All
// disruption mode.
func hasAllDisruptionMode(cfg *SchedulingConfig) bool {
	return cfg != nil && cfg.DisruptionMode != nil && cfg.DisruptionMode.All != nil
}

// hasSingleDisruptionMode reports whether the config resolves to the Single
// disruption mode, either explicitly or by defaulting when no mode is set.
func hasSingleDisruptionMode(cfg *SchedulingConfig) bool {
	if cfg == nil || cfg.DisruptionMode == nil || *cfg.DisruptionMode == (DisruptionMode{}) {
		return true
	}
	return cfg.DisruptionMode.Single != nil
}
