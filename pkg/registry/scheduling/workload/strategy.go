/*
Copyright 2025 The Kubernetes Authors.

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

package workload

import (
	"context"

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

// workloadStrategy implements behavior for Workload objects.
type workloadStrategy struct {
	rest.DeclarativeValidation
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating Workload objects.
var Strategy = workloadStrategy{rest.DeclarativeValidation{Scheme: legacyscheme.Scheme}, names.SimpleNameGenerator}

func (workloadStrategy) NamespaceScoped() bool {
	return true
}

func (workloadStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	dropDisabledWorkloadFields(obj.(*scheduling.Workload), nil)
}

func (workloadStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	workloadScheduling := obj.(*scheduling.Workload)
	return validation.ValidateWorkload(workloadScheduling)
}

// DeclarativeValidationConfig implements rest.DeclarativeValidationConfigurer to supply declarative
// validation options to the generic BeforeCreate/BeforeUpdate code path.
func (workloadStrategy) DeclarativeValidationConfig(ctx context.Context, obj, oldObj runtime.Object) rest.DeclarativeValidationConfig {
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
	return rest.DeclarativeValidationConfig{DeclarativeEnforcement: true, Options: opts}
}

func (workloadStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

func (workloadStrategy) Canonicalize(obj runtime.Object) {}

func (workloadStrategy) AllowCreateOnUpdate() bool {
	return false
}

func (workloadStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	dropDisabledWorkloadFields(obj.(*scheduling.Workload), old.(*scheduling.Workload))
}

func (workloadStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateWorkloadUpdate(obj.(*scheduling.Workload), old.(*scheduling.Workload))
}

func (workloadStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func (workloadStrategy) AllowUnconditionalUpdate() bool {
	return true
}

// dropDisabledWorkloadFields removes fields which are covered by a feature gate.
func dropDisabledWorkloadFields(workload, oldWorkload *scheduling.Workload) {
	var workloadSpec, oldWorkloadSpec *scheduling.WorkloadSpec
	if workload != nil {
		workloadSpec = &workload.Spec
	}
	if oldWorkload != nil {
		oldWorkloadSpec = &oldWorkload.Spec
	}
	dropDisabledWorkloadSpecFields(workloadSpec, oldWorkloadSpec)
}

func dropDisabledWorkloadSpecFields(workloadSpec, oldWorkloadSpec *scheduling.WorkloadSpec) {
	var templates, oldTemplates []scheduling.PodGroupTemplate
	if workloadSpec != nil {
		templates = workloadSpec.PodGroupTemplates
	}
	if oldWorkloadSpec != nil {
		oldTemplates = oldWorkloadSpec.PodGroupTemplates
	}
	dropDisabledPodGroupTemplatesFields(templates, oldTemplates)
}

func dropDisabledPodGroupTemplatesFields(templates, oldTemplates []scheduling.PodGroupTemplate) {
	m := len(oldTemplates)
	for i := range templates {
		var oldTemplate *scheduling.PodGroupTemplate
		if i < m {
			oldTemplate = &oldTemplates[i]
		}
		template := &templates[i]
		dropDisabledSchedulingConstraintsFields(template, oldTemplate)
		dropDisabledDRAWorkloadResourceClaimsFields(template, oldTemplate)
		dropDisabledDisruptionModeField(template, oldTemplate)
		dropDisabledPriorityClassNameField(template, oldTemplate)
		dropDisabledPriorityField(template, oldTemplate)
	}
}

// dropDisabledSchedulingConstraintsFields drops the SchedulingConstraints field
// from the PodGroupTemplate if the TopologyAwareWorkloadScheduling feature gate is disabled.
func dropDisabledSchedulingConstraintsFields(template, oldTemplate *scheduling.PodGroupTemplate) {
	if utilfeature.DefaultFeatureGate.Enabled(features.TopologyAwareWorkloadScheduling) || schedulingConstraintsInUse(oldTemplate) {
		return
	}
	template.SchedulingConstraints = nil
}

// dropDisabledDRAWorkloadResourceClaimsFields removes resource claim references from
// podGroupTemplates unless they are already used by the old Workload spec.
func dropDisabledDRAWorkloadResourceClaimsFields(template, oldTemplate *scheduling.PodGroupTemplate) {
	if utilfeature.DefaultFeatureGate.Enabled(features.DRAWorkloadResourceClaims) || draWorkloadResourceClaimsInUse(oldTemplate) {
		return
	}
	template.ResourceClaims = nil
}

// dropDisabledDisruptionModeField removes the DisruptionMode field from a template
// unless it is already used in the old template.
func dropDisabledDisruptionModeField(template, oldTemplate *scheduling.PodGroupTemplate) {
	if utilfeature.DefaultFeatureGate.Enabled(features.WorkloadAwarePreemption) || disruptionModeInUse(oldTemplate) {
		// No need to drop anything.
		return
	}
	template.DisruptionMode = nil
}

// dropDisabledPriorityClassNameField removes the PriorityClassName field from a template
// unless it is already used in the old template.
func dropDisabledPriorityClassNameField(template, oldTemplate *scheduling.PodGroupTemplate) {
	if utilfeature.DefaultFeatureGate.Enabled(features.WorkloadAwarePreemption) || priorityClassNameInUse(oldTemplate) {
		// No need to drop anything.
		return
	}
	template.PriorityClassName = ""
}

// dropDisabledPriorityField removes the Priority field from a template unless it is
// already used in the old template.
func dropDisabledPriorityField(template, oldTemplate *scheduling.PodGroupTemplate) {
	if utilfeature.DefaultFeatureGate.Enabled(features.WorkloadAwarePreemption) || priorityInUse(oldTemplate) {
		// No need to drop anything.
		return
	}
	template.Priority = nil
}

func schedulingConstraintsInUse(pgt *scheduling.PodGroupTemplate) bool {
	return pgt != nil && pgt.SchedulingConstraints != nil
}

func draWorkloadResourceClaimsInUse(pgt *scheduling.PodGroupTemplate) bool {
	return pgt != nil && len(pgt.ResourceClaims) > 0
}

func disruptionModeInUse(pgt *scheduling.PodGroupTemplate) bool {
	return pgt != nil && pgt.DisruptionMode != nil
}

func priorityClassNameInUse(pgt *scheduling.PodGroupTemplate) bool {
	return pgt != nil && pgt.PriorityClassName != ""
}

func priorityInUse(pgt *scheduling.PodGroupTemplate) bool {
	return pgt != nil && pgt.Priority != nil
}
