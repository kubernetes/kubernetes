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

package podgroup

import (
	"context"

	"sigs.k8s.io/structured-merge-diff/v6/fieldpath"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage/names"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/apis/scheduling/validation"
	"k8s.io/kubernetes/pkg/features"
)

// podGroupStrategy implements behavior for PodGroup objects.
type podGroupStrategy struct {
	rest.DeclarativeValidation
	names.NameGenerator
}

// NewStrategy is the default logic that applies when creating and updating PodGroup objects.
func NewStrategy() *podGroupStrategy {
	return &podGroupStrategy{
		rest.DeclarativeValidation{Scheme: legacyscheme.Scheme},
		names.SimpleNameGenerator,
	}
}

func (*podGroupStrategy) NamespaceScoped() bool {
	return true
}

// GetResetFields returns the set of fields that get reset by the strategy and
// should not be modified by the user. For a new PodGroup that is the status.
func (*podGroupStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"scheduling.k8s.io/v1alpha2": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
	}

	return fields
}

func (*podGroupStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	podGroup := obj.(*scheduling.PodGroup)
	// Status must not be set by user on create.
	podGroup.Status = scheduling.PodGroupStatus{}
	dropDisabledPodGroupFields(podGroup, nil)
}

func (*podGroupStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	podGroup := obj.(*scheduling.PodGroup)
	return validation.ValidatePodGroup(podGroup)
}

func (*podGroupStrategy) DeclarativeValidationConfig(ctx context.Context, obj, oldObj runtime.Object) rest.DeclarativeValidationConfig {
	opts := []string{}
	if utilfeature.DefaultFeatureGate.Enabled(features.TopologyAwareWorkloadScheduling) {
		opts = append(opts, string(features.TopologyAwareWorkloadScheduling))
	}
	if utilfeature.DefaultFeatureGate.Enabled(features.DRAWorkloadResourceClaims) {
		opts = append(opts, string(features.DRAWorkloadResourceClaims))
	}
	if utilfeature.DefaultFeatureGate.Enabled(features.WorkloadAwarePreemption) {
		opts = append(opts, string(features.WorkloadAwarePreemption))
	}
	return rest.DeclarativeValidationConfig{Options: opts, DeclarativeEnforcement: true}
}

func (*podGroupStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

func (*podGroupStrategy) Canonicalize(obj runtime.Object) {}

func (*podGroupStrategy) AllowCreateOnUpdate() bool {
	return false
}

func (*podGroupStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newPodGroup := obj.(*scheduling.PodGroup)
	oldPodGroup := old.(*scheduling.PodGroup)
	newPodGroup.Status = oldPodGroup.Status
	dropDisabledPodGroupFields(newPodGroup, oldPodGroup)
}

func (*podGroupStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	newPodGroup := obj.(*scheduling.PodGroup)
	oldPodGroup := old.(*scheduling.PodGroup)
	return validation.ValidatePodGroupUpdate(newPodGroup, oldPodGroup)
}

func (*podGroupStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func (*podGroupStrategy) AllowUnconditionalUpdate() bool {
	return true
}

type podGroupStatusStrategy struct {
	*podGroupStrategy
}

// NewStatusStrategy creates a strategy for operating the status object.
func NewStatusStrategy(podGroupStrategy *podGroupStrategy) *podGroupStatusStrategy {
	return &podGroupStatusStrategy{podGroupStrategy}
}

// GetResetFields returns the set of fields that get reset by the strategy and
// should not be modified by the user. For a status update that is the spec.
func (*podGroupStatusStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"scheduling.k8s.io/v1alpha2": fieldpath.NewSet(
			fieldpath.MakePathOrDie("metadata"),
			fieldpath.MakePathOrDie("spec"),
		),
	}

	return fields
}

func (*podGroupStatusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newPodGroup := obj.(*scheduling.PodGroup)
	oldPodGroup := old.(*scheduling.PodGroup)
	newPodGroup.Spec = oldPodGroup.Spec
	metav1.ResetObjectMetaForStatus(&newPodGroup.ObjectMeta, &oldPodGroup.ObjectMeta)
	dropDisabledPodGroupStatusFields(newPodGroup, oldPodGroup)
}

func (r *podGroupStatusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	newPodGroup := obj.(*scheduling.PodGroup)
	oldPodGroup := old.(*scheduling.PodGroup)
	return validation.ValidatePodGroupStatusUpdate(newPodGroup, oldPodGroup)
}

// WarningsOnUpdate returns warnings for the given update.
func (*podGroupStatusStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

// dropDisabledPodGroupFields removes fields which are covered by a feature gate.
func dropDisabledPodGroupFields(podGroup, oldPodGroup *scheduling.PodGroup) {
	var podGroupSpec, oldPodGroupSpec *scheduling.PodGroupSpec
	if podGroup != nil {
		podGroupSpec = &podGroup.Spec
	}
	if oldPodGroup != nil {
		oldPodGroupSpec = &oldPodGroup.Spec
	}
	dropDisabledPodGroupSpecFields(podGroupSpec, oldPodGroupSpec)
}

func dropDisabledPodGroupSpecFields(podGroupSpec, oldPodGroupSpec *scheduling.PodGroupSpec) {
	dropDisabledSchedulingConstraintsFields(podGroupSpec, oldPodGroupSpec)
	dropDisabledDRAWorkloadResourceClaimsFields(podGroupSpec, oldPodGroupSpec)
	dropDisabledDisruptionModeField(podGroupSpec, oldPodGroupSpec)
	dropDisabledPriorityClassNameField(podGroupSpec, oldPodGroupSpec)
	dropDisabledPriorityField(podGroupSpec, oldPodGroupSpec)
}

func dropDisabledPodGroupStatusFields(newPodGroup, oldPodGroup *scheduling.PodGroup) {
	var oldPodGroupSpec *scheduling.PodGroupSpec
	if oldPodGroup != nil {
		oldPodGroupSpec = &oldPodGroup.Spec
	}
	if utilfeature.DefaultFeatureGate.Enabled(features.DRAWorkloadResourceClaims) || draWorkloadResourceClaimsInUse(oldPodGroupSpec) {
		// No need to drop anything.
		return
	}
	newPodGroup.Status.ResourceClaimStatuses = nil
}

// dropDisabledSchedulingConstraintsFields drops the SchedulingConstraints field
// from the new PodGroup if the TopologyAwareWorkloadScheduling feature gate is disabled
// and it was not used in the old PodGroup.
func dropDisabledSchedulingConstraintsFields(podGroupSpec, oldPodGroupSpec *scheduling.PodGroupSpec) {
	if utilfeature.DefaultFeatureGate.Enabled(features.TopologyAwareWorkloadScheduling) || schedulingConstraintsInUse(oldPodGroupSpec) {
		// No need to drop anything.
		return
	}
	podGroupSpec.SchedulingConstraints = nil
}

// dropDisabledDRAWorkloadResourceClaimsFields removes resource claim references
// unless they are already used by the old PodGroup spec.
func dropDisabledDRAWorkloadResourceClaimsFields(podGroupSpec, oldPodGroupSpec *scheduling.PodGroupSpec) {
	if utilfeature.DefaultFeatureGate.Enabled(features.DRAWorkloadResourceClaims) || draWorkloadResourceClaimsInUse(oldPodGroupSpec) {
		// No need to drop anything.
		return
	}
	podGroupSpec.ResourceClaims = nil
}

// dropDisabledDisruptionModeField removes the DisruptionMode field unless it is
// already used in the old PodGroup spec.
func dropDisabledDisruptionModeField(podGroupSpec, oldPodGroupSpec *scheduling.PodGroupSpec) {
	if utilfeature.DefaultFeatureGate.Enabled(features.WorkloadAwarePreemption) || disruptionModeInUse(oldPodGroupSpec) {
		// No need to drop anything.
		return
	}
	podGroupSpec.DisruptionMode = nil
}

// dropDisabledPriorityClassNameField removes the PriorityClassName field unless
// it is already used in the old PodGroup spec.
func dropDisabledPriorityClassNameField(podGroupSpec, oldPodGroupSpec *scheduling.PodGroupSpec) {
	if utilfeature.DefaultFeatureGate.Enabled(features.WorkloadAwarePreemption) || priorityClassNameInUse(oldPodGroupSpec) {
		// No need to drop anything.
		return
	}
	podGroupSpec.PriorityClassName = ""
}

// dropDisabledPriorityField removes the Priority field unless it is already used
// in the old PodGroup spec.
func dropDisabledPriorityField(podGroupSpec, oldPodGroupSpec *scheduling.PodGroupSpec) {
	if utilfeature.DefaultFeatureGate.Enabled(features.WorkloadAwarePreemption) || priorityInUse(oldPodGroupSpec) {
		// No need to drop anything.
		return
	}
	podGroupSpec.Priority = nil
}

func schedulingConstraintsInUse(podGroupSpec *scheduling.PodGroupSpec) bool {
	return podGroupSpec != nil && podGroupSpec.SchedulingConstraints != nil
}

func draWorkloadResourceClaimsInUse(podGroupSpec *scheduling.PodGroupSpec) bool {
	return podGroupSpec != nil && len(podGroupSpec.ResourceClaims) > 0
}

func disruptionModeInUse(podGroupSpec *scheduling.PodGroupSpec) bool {
	return podGroupSpec != nil && podGroupSpec.DisruptionMode != nil
}

func priorityClassNameInUse(podGroupSpec *scheduling.PodGroupSpec) bool {
	return podGroupSpec != nil && podGroupSpec.PriorityClassName != ""
}

func priorityInUse(podGroupSpec *scheduling.PodGroupSpec) bool {
	return podGroupSpec != nil && podGroupSpec.Priority != nil
}
