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

// This file is the public surface of the workloadbuilder library: the methods
// controllers call. The Builder type and its options live in builder.go.

package workloadbuilder

import (
	"context"
	"fmt"

	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	"k8s.io/apimachinery/pkg/api/operation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// NewBuilder returns a Builder for the given WorkloadItem tree and options.
func NewBuilder(root *WorkloadItem, opts BuildOptions) *Builder {
	return &Builder{root: root, opts: opts}
}

// SetExistingWorkload puts the Builder in materialize-from-existing mode: it uses
// the given already-compiled Workload as the source for NewPodGroup instead of a
// Workload compiled from the item tree. BuildWorkload is refused afterwards so the
// supplied Workload is never recompiled over.
func (b *Builder) SetExistingWorkload(workload *schedulingv1alpha3.Workload) {
	b.workload = workload
	b.externalWorkload = true
}

// Validate runs declarative validation on the input blocks (unless disabled)
// and controller-policy checks that declarative validation cannot express.
// Validate runs declarative validation on the input blocks, as well as
// controller-policy checks that declarative validation cannot express.
// For create operations, pass operation.Create and nil for oldRoot.
// For update operations, pass operation.Update and the previous WorkloadItem.
func (b *Builder) Validate(ctx context.Context, op operation.Operation, oldRoot *WorkloadItem, rootPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList

	if b.root == nil {
		return field.ErrorList{field.Invalid(field.NewPath(""), nil, "invalid builder: compile: missing root WorkloadItem")}
	}

	if rootPath == nil {
		rootPath = field.NewPath("")
	}

	if !b.opts.DisableDeclarativeValidation {
		allErrs = append(allErrs, b.validateDeclarative(ctx, op, oldRoot, rootPath)...)
	}

	resolvedConfig := resolveSchedulingConfig(b.root)
	allErrs = append(allErrs, b.validateAllowedSchedulingPolicies(b.root, resolvedConfig, rootPath)...)
	allErrs = append(allErrs, b.validateAllowedDisruptionModes(b.root, resolvedConfig, rootPath)...)
	allErrs = append(allErrs, b.validateDisruptionModeCompatibleWithSchedulingPolicy(b.root, resolvedConfig, rootPath)...)

	return allErrs
}




// appendPathElements appends the given elements to the root path.
// If elements is empty, it returns the root path itself.
func appendPathElements(rootPath *field.Path, elements []string) *field.Path {
	path := rootPath
	for _, el := range elements {
		path = path.Child(el)
	}
	return path
}

func (b *Builder) validateDeclarative(ctx context.Context, op operation.Operation, oldRoot *WorkloadItem, rootPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList

	if b.root.Input.Policy.PodGroupData != nil || (oldRoot != nil && oldRoot.Input.Policy.PodGroupData != nil) {
		path := appendPathElements(rootPath, b.root.Input.Policy.PathElements)
		var oldData *schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy
		if oldRoot != nil && oldRoot.Input.Policy.PodGroupData != nil {
			oldData = oldRoot.Input.Policy.PodGroupData
		}
		allErrs = append(allErrs, schedulingv1alpha3.Validate_WorkloadPodGroupSchedulingPolicy(ctx, op, path, b.root.Input.Policy.PodGroupData, oldData)...)
	}
	if b.root.Input.Constraints.PodGroupData != nil || (oldRoot != nil && oldRoot.Input.Constraints.PodGroupData != nil) {
		path := appendPathElements(rootPath, b.root.Input.Constraints.PathElements)
		var oldData *schedulingv1alpha3.WorkloadPodGroupSchedulingConstraints
		if oldRoot != nil && oldRoot.Input.Constraints.PodGroupData != nil {
			oldData = oldRoot.Input.Constraints.PodGroupData
		}
		allErrs = append(allErrs, schedulingv1alpha3.Validate_WorkloadPodGroupSchedulingConstraints(ctx, op, path, b.root.Input.Constraints.PodGroupData, oldData)...)
	}
	if b.root.Input.DisruptionMode.PodGroupData != nil || (oldRoot != nil && oldRoot.Input.DisruptionMode.PodGroupData != nil) {
		path := appendPathElements(rootPath, b.root.Input.DisruptionMode.PathElements)
		var oldData *schedulingv1alpha3.WorkloadPodGroupDisruptionMode
		if oldRoot != nil && oldRoot.Input.DisruptionMode.PodGroupData != nil {
			oldData = oldRoot.Input.DisruptionMode.PodGroupData
		}
		allErrs = append(allErrs, schedulingv1alpha3.Validate_WorkloadPodGroupDisruptionMode(ctx, op, path, b.root.Input.DisruptionMode.PodGroupData, oldData)...)
	}
	
	newLen := len(b.root.Input.ResourceClaims.PodGroupData)
	oldLen := 0
	if oldRoot != nil {
		oldLen = len(oldRoot.Input.ResourceClaims.PodGroupData)
	}
	if newLen > 0 || oldLen > 0 {
		var path *field.Path
		if len(b.root.Input.ResourceClaims.PathElements) > 0 {
			if rootPath != nil {
				path = rootPath.Child(b.root.Input.ResourceClaims.PathElements[0], b.root.Input.ResourceClaims.PathElements[1:]...)
			} else {
				path = field.NewPath(b.root.Input.ResourceClaims.PathElements[0], b.root.Input.ResourceClaims.PathElements[1:]...)
			}
		} else if rootPath != nil {
			path = rootPath.Child("resourceClaims")
		} else {
			path = field.NewPath("resourceClaims")
		}
		maxLen := newLen
		if oldLen > maxLen {
			maxLen = oldLen
		}
		for idx := 0; idx < maxLen; idx++ {
			var data, oldData *schedulingv1alpha3.WorkloadPodGroupResourceClaim
			if idx < newLen {
				data = &b.root.Input.ResourceClaims.PodGroupData[idx]
			}
			if idx < oldLen {
				oldData = &oldRoot.Input.ResourceClaims.PodGroupData[idx]
			}
			allErrs = append(allErrs, schedulingv1alpha3.Validate_WorkloadPodGroupResourceClaim(ctx, op, path.Index(idx), data, oldData)...)
		}
	}

	return allErrs
}
// BuildWorkload compiles the WorkloadItem tree into a Workload, sets its
// identity and controllerRef, and caches the result.
func (b *Builder) BuildWorkload() (*schedulingv1alpha3.Workload, error) {
	if b.externalWorkload {
		return nil, fmt.Errorf("BuildWorkload cannot be used after SetExistingWorkload: the Builder materializes from a supplied Workload")
	}
	if b.workload != nil {
		return b.workload, nil
	}
	wl, err := b.build()
	if err != nil {
		return nil, err
	}
	b.workload = wl
	return wl, nil
}

// NewPodGroup materializes a runtime PodGroup from the named PodGroupTemplate of
// the Builder's Workload. The Workload must be available first: either compiled
// via BuildWorkload or supplied via SetExistingWorkload. The PodGroup is owned by
// the Builder's configured Owner (if any); callers needing additional owner
// references append them to the returned object.
func (b *Builder) NewPodGroup(podGroupName, templateName string) (*schedulingv1alpha3.PodGroup, error) {
	if b.workload == nil {
		return nil, fmt.Errorf("no Workload available for NewPodGroup: call BuildWorkload or SetExistingWorkload first")
	}
	var owners []metav1.OwnerReference
	if b.opts.Owner != nil {
		owners = []metav1.OwnerReference{*b.opts.Owner}
	}
	return newPodGroup(b.workload, templateName, podGroupName, owners)
}
