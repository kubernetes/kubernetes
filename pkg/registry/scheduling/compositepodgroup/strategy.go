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

	"k8s.io/apimachinery/pkg/api/operation"
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

// compositePodGroupStrategy implements behavior for CompositePodGroup objects.
type compositePodGroupStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// NewStrategy is the default logic that applies when creating and updating CompositePodGroup objects.
func NewStrategy() *compositePodGroupStrategy {
	return &compositePodGroupStrategy{
		legacyscheme.Scheme,
		names.SimpleNameGenerator,
	}
}

func (*compositePodGroupStrategy) NamespaceScoped() bool {
	return true
}

// GetResetFields returns the set of fields that get reset by the strategy and
// should not be modified by the user. For a new CompositePodGroup that is the status.
func (*compositePodGroupStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"scheduling.k8s.io/v1alpha2": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
	}

	return fields
}

func (*compositePodGroupStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	compositePodGroup := obj.(*scheduling.CompositePodGroup)
	// Status must not be set by user on create.
	compositePodGroup.Status = scheduling.CompositePodGroupStatus{}
	dropDisabledCompositePodGroupFields(compositePodGroup, nil)
}

func (*compositePodGroupStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	compositePodGroup := obj.(*scheduling.CompositePodGroup)
	allErrs := validation.ValidateCompositePodGroup(compositePodGroup)
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
	if utilfeature.DefaultFeatureGate.Enabled(features.CompositePodGroup) {
		opts = append(opts, string(features.CompositePodGroup))
	}
	return rest.ValidateDeclarativelyWithMigrationChecks(ctx, legacyscheme.Scheme, obj, nil, allErrs, operation.Create, rest.WithDeclarativeEnforcement(), rest.WithOptions(opts))
}

func (*compositePodGroupStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

func (*compositePodGroupStrategy) Canonicalize(obj runtime.Object) {}

func (*compositePodGroupStrategy) AllowCreateOnUpdate() bool {
	return false
}

func (*compositePodGroupStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newCompositePodGroup := obj.(*scheduling.CompositePodGroup)
	oldCompositePodGroup := old.(*scheduling.CompositePodGroup)
	newCompositePodGroup.Status = oldCompositePodGroup.Status
	dropDisabledCompositePodGroupFields(newCompositePodGroup, oldCompositePodGroup)
}

func (*compositePodGroupStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	newCompositePodGroup := obj.(*scheduling.CompositePodGroup)
	oldCompositePodGroup := old.(*scheduling.CompositePodGroup)
	allErrs := validation.ValidateCompositePodGroupUpdate(newCompositePodGroup, oldCompositePodGroup)
	opts := []string{}
	// Declarative validation will always allow fields to remain unchanged, so if any
	// of the fields which are covered by these gates are set, we will not re-validate them
	// (even if the gates are disabled) as long as they do not change values. If a gate
	// is disabled, they will not be allowed to change values.
	if utilfeature.DefaultFeatureGate.Enabled(features.TopologyAwareWorkloadScheduling) {
		opts = append(opts, string(features.TopologyAwareWorkloadScheduling))
	}
	if utilfeature.DefaultFeatureGate.Enabled(features.DRAWorkloadResourceClaims) {
		opts = append(opts, string(features.DRAWorkloadResourceClaims))
	}
	if utilfeature.DefaultFeatureGate.Enabled(features.WorkloadAwarePreemption) {
		opts = append(opts, string(features.WorkloadAwarePreemption))
	}
	if utilfeature.DefaultFeatureGate.Enabled(features.CompositePodGroup) {
		opts = append(opts, string(features.CompositePodGroup))
	}
	return rest.ValidateDeclarativelyWithMigrationChecks(ctx, legacyscheme.Scheme, obj, old, allErrs, operation.Update, rest.WithDeclarativeEnforcement(), rest.WithOptions(opts))
}

func (*compositePodGroupStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func (*compositePodGroupStrategy) AllowUnconditionalUpdate() bool {
	return true
}

type compositePodGroupStatusStrategy struct {
	*compositePodGroupStrategy
}

// NewStatusStrategy creates a strategy for operating the status object.
func NewStatusStrategy(compositePodGroupStrategy *compositePodGroupStrategy) *compositePodGroupStatusStrategy {
	return &compositePodGroupStatusStrategy{compositePodGroupStrategy}
}

// GetResetFields returns the set of fields that get reset by the strategy and
// should not be modified by the user. For a status update that is the spec.
func (*compositePodGroupStatusStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"scheduling.k8s.io/v1alpha2": fieldpath.NewSet(
			fieldpath.MakePathOrDie("metadata"),
			fieldpath.MakePathOrDie("spec"),
		),
	}

	return fields
}

func (*compositePodGroupStatusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newCompositePodGroup := obj.(*scheduling.CompositePodGroup)
	oldCompositePodGroup := old.(*scheduling.CompositePodGroup)
	newCompositePodGroup.Spec = oldCompositePodGroup.Spec
	metav1.ResetObjectMetaForStatus(&newCompositePodGroup.ObjectMeta, &oldCompositePodGroup.ObjectMeta)
	dropDisabledCompositePodGroupStatusFields(newCompositePodGroup, oldCompositePodGroup)
}

func (*compositePodGroupStatusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	newCompositePodGroup := obj.(*scheduling.CompositePodGroup)
	oldCompositePodGroup := old.(*scheduling.CompositePodGroup)
	errs := validation.ValidateCompositePodGroupStatusUpdate(newCompositePodGroup, oldCompositePodGroup)
	return rest.ValidateDeclarativelyWithMigrationChecks(ctx, legacyscheme.Scheme, obj, old, errs, operation.Update, rest.WithDeclarativeEnforcement())
}

// WarningsOnUpdate returns warnings for the given update.
func (*compositePodGroupStatusStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

// dropDisabledCompositePodGroupFields removes fields which are covered by a feature gate.
func dropDisabledCompositePodGroupFields(compositePodGroup, oldCompositePodGroup *scheduling.CompositePodGroup) {
	var compositePodGroupSpec, oldCompositePodGroupSpec *scheduling.CompositePodGroupSpec
	if compositePodGroup != nil {
		compositePodGroupSpec = &compositePodGroup.Spec
	}
	if oldCompositePodGroup != nil {
		oldCompositePodGroupSpec = &oldCompositePodGroup.Spec
	}
	dropDisabledCompositePodGroupSpecFields(compositePodGroupSpec, oldCompositePodGroupSpec)
}

func dropDisabledCompositePodGroupSpecFields(compositePodGroupSpec, oldCompositePodGroupSpec *scheduling.CompositePodGroupSpec) {
}

func dropDisabledCompositePodGroupStatusFields(newCompositePodGroup, oldCompositePodGroup *scheduling.CompositePodGroup) {
}
