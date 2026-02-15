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

	"k8s.io/apimachinery/pkg/api/operation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/apis/scheduling/validation"
)

// podGroupStrategy implements behavior for PodGroup objects.
type podGroupStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// NewStrategy is the default logic that applies when creating and updating PodGroup objects.
func NewStrategy() *podGroupStrategy {
	return &podGroupStrategy{
		legacyscheme.Scheme,
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
}

func (*podGroupStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	podGroup := obj.(*scheduling.PodGroup)
	allErrs := validation.ValidatePodGroup(podGroup)
	return rest.ValidateDeclarativelyWithMigrationChecks(ctx, legacyscheme.Scheme, obj, nil, allErrs, operation.Create)
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
}

func (*podGroupStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	newPodGroup := obj.(*scheduling.PodGroup)
	oldPodGroup := old.(*scheduling.PodGroup)
	allErrs := validation.ValidatePodGroupUpdate(newPodGroup, oldPodGroup)
	return rest.ValidateDeclarativelyWithMigrationChecks(ctx, legacyscheme.Scheme, newPodGroup, oldPodGroup, allErrs, operation.Update)
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
}

func (r *podGroupStatusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	newPodGroup := obj.(*scheduling.PodGroup)
	oldPodGroup := old.(*scheduling.PodGroup)
	errs := validation.ValidatePodGroupStatusUpdate(newPodGroup, oldPodGroup)
	return rest.ValidateDeclarativelyWithMigrationChecks(ctx, legacyscheme.Scheme, oldPodGroup, oldPodGroup, errs, operation.Update)
}

// WarningsOnUpdate returns warnings for the given update.
func (*podGroupStatusStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}
