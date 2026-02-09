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
	return rest.ValidateDeclarativelyWithMigrationChecks(ctx, legacyscheme.Scheme, obj, nil, allErrs, operation.Create)
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
	return rest.ValidateDeclarativelyWithMigrationChecks(ctx, legacyscheme.Scheme, obj, old, allErrs, operation.Update)
}

func (workloadStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func (workloadStrategy) AllowUnconditionalUpdate() bool {
	return true
}

// dropDisabledFields removes fields which are covered by a feature gate.
func dropDisabledFields(newWorkload, oldWorkload *scheduling.Workload) {
	dropDisabledDisruptionModeFields(newWorkload, oldWorkload)
}

// dropDisabledDisruptionModeFields drops DisruptionMode fields from the new workload
// if they were not used in the old workload.
func dropDisabledDisruptionModeFields(newWorkload, oldWorkload *scheduling.Workload) {
	if utilfeature.DefaultFeatureGate.Enabled(features.WorkloadAwarePreemption) || disruptionModeInUse(oldWorkload) {
		// No need to drop anything.
		return
	}
	for _, pg := range newWorkload.Spec.PodGroups {
		if pg.Policy.Gang != nil {
			pg.Policy.Gang.DisruptionMode = nil
		}
	}
}

func disruptionModeInUse(workload *scheduling.Workload) bool {
	if workload == nil {
		return false
	}
	for _, pg := range workload.Spec.PodGroups {
		if pg.Policy.Gang != nil && pg.Policy.Gang.DisruptionMode != nil {
			return true
		}
	}
	return false
}
