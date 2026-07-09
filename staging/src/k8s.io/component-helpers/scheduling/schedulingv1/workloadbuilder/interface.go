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
	"fmt"

	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
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

// Validate runs the controller-policy checks that declarative validation
// cannot express. It compiles the Workload and checks the result: every
// PodGroupTemplate's scheduling policy and disruption mode must be in the
// Builder's allow-lists and no template may combine the Basic policy with All
// disruption. It deliberately does not re-run the structural checks that
// compiling performs, as the in-tree controllers get those from generated
// declarative validation.
func (b *Builder) Validate(fldPath *field.Path) field.ErrorList {
	workload, err := b.build()
	if err != nil {
		return field.ErrorList{field.Invalid(fldPath, "", err.Error())}
	}

	var allErrs field.ErrorList
	allErrs = append(allErrs, b.validateAllowedSchedulingPolicies(fldPath, workload)...)
	allErrs = append(allErrs, b.validateAllowedDisruptionModes(fldPath, workload)...)
	allErrs = append(allErrs, b.validateDisruptionModeCompatibleWithSchedulingPolicy(fldPath, workload)...)
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
