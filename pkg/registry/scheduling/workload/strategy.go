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
	"slices"

	"k8s.io/apimachinery/pkg/api/operation"
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
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating Workload objects.
var Strategy = workloadStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

func (workloadStrategy) NamespaceScoped() bool {
	return true
}

func (workloadStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	workload := obj.(*scheduling.Workload)
	dropDisabledFields(workload, nil)
}

func (workloadStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	workloadScheduling := obj.(*scheduling.Workload)
	allErrs := validation.ValidateWorkload(workloadScheduling)
	return rest.ValidateDeclarativelyWithMigrationChecks(ctx, legacyscheme.Scheme, obj, nil, allErrs, operation.Create, rest.WithDeclarativeEnforcement())
}

func (workloadStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

func (workloadStrategy) Canonicalize(obj runtime.Object) {}

func (workloadStrategy) AllowCreateOnUpdate() bool {
	return false
}

func (workloadStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newWorkload := obj.(*scheduling.Workload)
	oldWorkload := old.(*scheduling.Workload)
	dropDisabledFields(newWorkload, oldWorkload)
}

func (workloadStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	allErrs := validation.ValidateWorkloadUpdate(obj.(*scheduling.Workload), old.(*scheduling.Workload))
	return rest.ValidateDeclarativelyWithMigrationChecks(ctx, legacyscheme.Scheme, obj, old, allErrs, operation.Update, rest.WithDeclarativeEnforcement())
}

func (workloadStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func (workloadStrategy) AllowUnconditionalUpdate() bool {
	return true
}

func dropDisabledFields(workload, oldWorkload *scheduling.Workload) {
	var workloadSpec, oldWorkloadSpec *scheduling.WorkloadSpec
	if workload != nil {
		workloadSpec = &workload.Spec
	}
	if oldWorkload != nil {
		oldWorkloadSpec = &oldWorkload.Spec
	}
	dropDisabledSpecFields(workloadSpec, oldWorkloadSpec)
}

func dropDisabledSpecFields(workloadSpec, oldWorkloadSpec *scheduling.WorkloadSpec) {
	dropDisabledDRAWorkloadResourceClaimsFields(workloadSpec, oldWorkloadSpec)
}

// dropDisabledDRAWorkloadResourceClaimsFields removes resource claim references from
// podGroupTemplates unless they are already used by the old Workload spec.
func dropDisabledDRAWorkloadResourceClaimsFields(workloadSpec, oldWorkloadSpec *scheduling.WorkloadSpec) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.DRAWorkloadResourceClaims) && !draWorkloadResourceClaimsInUse(oldWorkloadSpec) {
		for i := range workloadSpec.PodGroupTemplates {
			workloadSpec.PodGroupTemplates[i].ResourceClaims = nil
		}
	}
}

func draWorkloadResourceClaimsInUse(workloadSpec *scheduling.WorkloadSpec) bool {
	return workloadSpec != nil &&
		slices.ContainsFunc(workloadSpec.PodGroupTemplates, func(p scheduling.PodGroupTemplate) bool {
			return len(p.ResourceClaims) > 0
		})
}
