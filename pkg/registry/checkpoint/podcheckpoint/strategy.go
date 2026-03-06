/*
Copyright 2026 The Kubernetes Authors.

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

package podcheckpoint

import (
	"context"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/checkpoint"
	"k8s.io/kubernetes/pkg/apis/checkpoint/validation"
	"sigs.k8s.io/structured-merge-diff/v6/fieldpath"
)

// podCheckpointStrategy implements verification logic for PodCheckpoints.
type podCheckpointStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating PodCheckpoint objects.
var Strategy = podCheckpointStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

// NamespaceScoped returns true because PodCheckpoints are namespaced.
func (podCheckpointStrategy) NamespaceScoped() bool {
	return true
}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (podCheckpointStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"checkpoint.k8s.io/v1alpha1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
	}
	return fields
}

// PrepareForCreate clears the status of a PodCheckpoint before creation.
func (podCheckpointStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	pc := obj.(*checkpoint.PodCheckpoint)
	pc.Status = checkpoint.PodCheckpointStatus{
		Phase: checkpoint.PodCheckpointPending,
	}
	pc.Generation = 1
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (podCheckpointStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newPC := obj.(*checkpoint.PodCheckpoint)
	oldPC := old.(*checkpoint.PodCheckpoint)

	// Status is not allowed to be set by end users on update of the main resource.
	newPC.Status = oldPC.Status
}

// Validate validates a new PodCheckpoint.
func (podCheckpointStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	pc := obj.(*checkpoint.PodCheckpoint)
	return validation.ValidatePodCheckpoint(pc)
}

// WarningsOnCreate returns warnings for the creation of the given object.
func (podCheckpointStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

// Canonicalize normalizes the object after validation.
func (podCheckpointStrategy) Canonicalize(obj runtime.Object) {
}

// AllowCreateOnUpdate is false for PodCheckpoints.
func (podCheckpointStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (podCheckpointStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidatePodCheckpointUpdate(obj.(*checkpoint.PodCheckpoint), old.(*checkpoint.PodCheckpoint))
}

// WarningsOnUpdate returns warnings for the given update.
func (podCheckpointStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

// AllowUnconditionalUpdate is the default update policy for PodCheckpoint objects.
func (podCheckpointStrategy) AllowUnconditionalUpdate() bool {
	return false
}

// podCheckpointStatusStrategy implements verification logic for status updates of PodCheckpoints.
type podCheckpointStatusStrategy struct {
	podCheckpointStrategy
}

// StatusStrategy is the logic for status updates of PodCheckpoints.
var StatusStrategy = podCheckpointStatusStrategy{Strategy}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (podCheckpointStatusStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"checkpoint.k8s.io/v1alpha1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("spec"),
		),
	}
	return fields
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update of status.
func (podCheckpointStatusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newPC := obj.(*checkpoint.PodCheckpoint)
	oldPC := old.(*checkpoint.PodCheckpoint)
	// Spec is not allowed to be changed on status update.
	newPC.Spec = oldPC.Spec
}

// ValidateUpdate validates the update of status of a PodCheckpoint.
func (podCheckpointStatusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return field.ErrorList{}
}
