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

package compositepodgroup

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

type compositePodGroupStrategy struct {
	rest.DeclarativeValidation
	names.NameGenerator
}

func NewStrategy() *compositePodGroupStrategy {
	return &compositePodGroupStrategy{
		rest.DeclarativeValidation{Scheme: legacyscheme.Scheme},
		names.SimpleNameGenerator,
	}
}

func (*compositePodGroupStrategy) NamespaceScoped() bool {
	return true
}

func (*compositePodGroupStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"scheduling.k8s.io/v1alpha3": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
	}
	return fields
}

func (*compositePodGroupStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	cpg := obj.(*scheduling.CompositePodGroup)
	cpg.Status = scheduling.CompositePodGroupStatus{}
	dropDisabledCompositePodGroupFields(cpg, nil)
}

func (*compositePodGroupStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	cpg := obj.(*scheduling.CompositePodGroup)
	return validation.ValidateCompositePodGroup(cpg)
}

func (*compositePodGroupStrategy) DeclarativeValidationConfig(ctx context.Context, obj, oldObj runtime.Object) rest.DeclarativeValidationConfig {
	opts := []string{}
	if utilfeature.DefaultFeatureGate.Enabled(features.TopologyAwareWorkloadScheduling) {
		opts = append(opts, string(features.TopologyAwareWorkloadScheduling))
	}
	if utilfeature.DefaultFeatureGate.Enabled(features.WorkloadAwarePreemption) {
		opts = append(opts, string(features.WorkloadAwarePreemption))
	}
	if utilfeature.DefaultFeatureGate.Enabled(features.CompositePodGroup) {
		opts = append(opts, string(features.CompositePodGroup))
	}
	return rest.DeclarativeValidationConfig{Options: opts, DeclarativeEnforcement: true}
}

func (*compositePodGroupStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

func (*compositePodGroupStrategy) Canonicalize(obj runtime.Object) {}

func (*compositePodGroupStrategy) AllowCreateOnUpdate(ctx context.Context) bool {
	return false
}

func (*compositePodGroupStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newCPG := obj.(*scheduling.CompositePodGroup)
	oldCPG := old.(*scheduling.CompositePodGroup)
	newCPG.Status = oldCPG.Status
	dropDisabledCompositePodGroupFields(newCPG, oldCPG)
}

func (*compositePodGroupStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	newCPG := obj.(*scheduling.CompositePodGroup)
	oldCPG := old.(*scheduling.CompositePodGroup)
	return validation.ValidateCompositePodGroupUpdate(newCPG, oldCPG)
}

func (*compositePodGroupStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func (*compositePodGroupStrategy) AllowUnconditionalUpdate(ctx context.Context) bool {
	return true
}

type compositePodGroupStatusStrategy struct {
	*compositePodGroupStrategy
}

func NewStatusStrategy(strategy *compositePodGroupStrategy) *compositePodGroupStatusStrategy {
	return &compositePodGroupStatusStrategy{strategy}
}

func (*compositePodGroupStatusStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"scheduling.k8s.io/v1alpha3": fieldpath.NewSet(
			fieldpath.MakePathOrDie("metadata"),
			fieldpath.MakePathOrDie("spec"),
		),
	}
	return fields
}

func (*compositePodGroupStatusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newCPG := obj.(*scheduling.CompositePodGroup)
	oldCPG := old.(*scheduling.CompositePodGroup)
	newCPG.Spec = oldCPG.Spec
	metav1.ResetObjectMetaForStatus(&newCPG.ObjectMeta, &oldCPG.ObjectMeta)
}

func (r *compositePodGroupStatusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	newCPG := obj.(*scheduling.CompositePodGroup)
	oldCPG := old.(*scheduling.CompositePodGroup)
	return validation.ValidateCompositePodGroupStatusUpdate(newCPG, oldCPG)
}

func (*compositePodGroupStatusStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func dropDisabledCompositePodGroupFields(cpg, oldCPG *scheduling.CompositePodGroup) {
	var cpgSpec, oldCPGSpec *scheduling.CompositePodGroupSpec
	if cpg != nil {
		cpgSpec = &cpg.Spec
	}
	if oldCPG != nil {
		oldCPGSpec = &oldCPG.Spec
	}
	dropDisabledCompositePodGroupSpecFields(cpgSpec, oldCPGSpec)
}

func dropDisabledCompositePodGroupSpecFields(spec, oldSpec *scheduling.CompositePodGroupSpec) {
	dropDisabledCPGSchedulingConstraintsFields(spec, oldSpec)
	dropDisabledCPGDisruptionModeField(spec, oldSpec)
	dropDisabledCPGPriorityClassNameField(spec, oldSpec)
	dropDisabledCPGPriorityField(spec, oldSpec)
	dropDisabledCPGParentField(spec, oldSpec)
}

func dropDisabledCPGSchedulingConstraintsFields(spec, oldSpec *scheduling.CompositePodGroupSpec) {
	if utilfeature.DefaultFeatureGate.Enabled(features.TopologyAwareWorkloadScheduling) || cpgSchedulingConstraintsInUse(oldSpec) {
		return
	}
	spec.SchedulingConstraints = nil
}

func dropDisabledCPGDisruptionModeField(spec, oldSpec *scheduling.CompositePodGroupSpec) {
	if utilfeature.DefaultFeatureGate.Enabled(features.WorkloadAwarePreemption) || cpgDisruptionModeInUse(oldSpec) {
		return
	}
	spec.DisruptionMode = nil
}

func dropDisabledCPGPriorityClassNameField(spec, oldSpec *scheduling.CompositePodGroupSpec) {
	if utilfeature.DefaultFeatureGate.Enabled(features.WorkloadAwarePreemption) || cpgPriorityClassNameInUse(oldSpec) {
		return
	}
	spec.PriorityClassName = ""
}

func dropDisabledCPGPriorityField(spec, oldSpec *scheduling.CompositePodGroupSpec) {
	if utilfeature.DefaultFeatureGate.Enabled(features.WorkloadAwarePreemption) || cpgPriorityInUse(oldSpec) {
		return
	}
	spec.Priority = nil
}

func dropDisabledCPGParentField(spec, oldSpec *scheduling.CompositePodGroupSpec) {
	if utilfeature.DefaultFeatureGate.Enabled(features.CompositePodGroup) || cpgParentInUse(oldSpec) {
		return
	}
	spec.ParentCompositePodGroupName = nil
}

func cpgSchedulingConstraintsInUse(spec *scheduling.CompositePodGroupSpec) bool {
	return spec != nil && spec.SchedulingConstraints != nil
}

func cpgDisruptionModeInUse(spec *scheduling.CompositePodGroupSpec) bool {
	return spec != nil && spec.DisruptionMode != nil
}

func cpgPriorityClassNameInUse(spec *scheduling.CompositePodGroupSpec) bool {
	return spec != nil && spec.PriorityClassName != ""
}

func cpgPriorityInUse(spec *scheduling.CompositePodGroupSpec) bool {
	return spec != nil && spec.Priority != nil
}

func cpgParentInUse(spec *scheduling.CompositePodGroupSpec) bool {
	return spec != nil && spec.ParentCompositePodGroupName != nil
}
