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

package evictionrequest

import (
	"context"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/structured-merge-diff/v6/fieldpath"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/coordination"
	"k8s.io/kubernetes/pkg/apis/coordination/validation"
)

// evictionRequestStrategy is the default logic that applies when creating and updating EvictionRequest objects.
type evictionRequestStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

func NewStrategy() *evictionRequestStrategy {
	return &evictionRequestStrategy{
		legacyscheme.Scheme,
		names.SimpleNameGenerator,
	}
}

var _ = rest.ResetFieldsStrategy(&evictionRequestStrategy{})

func (*evictionRequestStrategy) NamespaceScoped() bool {
	return true
}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (*evictionRequestStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"coordination/v1alpha1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
	}
	return fields
}

// PrepareForCreate clears fields that are not allowed to be set by end users on creation.
func (*evictionRequestStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	evictionRequest := obj.(*coordination.EvictionRequest)
	evictionRequest.Status = coordination.EvictionRequestStatus{}
	evictionRequest.Generation = 1
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (*evictionRequestStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	oldEvictionRequest := old.(*coordination.EvictionRequest)
	newEvictionRequest := obj.(*coordination.EvictionRequest)
	newEvictionRequest.Status = oldEvictionRequest.Status

	// Spec updates bump the generation.
	if !apiequality.Semantic.DeepEqual(oldEvictionRequest.Spec, newEvictionRequest.Spec) {
		newEvictionRequest.Generation = oldEvictionRequest.Generation + 1
	}
}

// Validate validates a new EvictionRequest.
func (s *evictionRequestStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	evictionRequest := obj.(*coordination.EvictionRequest)

	allErrs := validation.ValidateEvictionRequest(evictionRequest)
	return rest.ValidateDeclarativelyWithMigrationChecks(ctx, legacyscheme.Scheme, obj, nil, allErrs, operation.Create, rest.DeclarativeValidationConfig{})
}
func (*evictionRequestStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

func (*evictionRequestStrategy) Canonicalize(obj runtime.Object) {
}

func (*evictionRequestStrategy) AllowCreateOnUpdate(ctx context.Context) bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (s *evictionRequestStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	var allErrs field.ErrorList
	evictionRequest := obj.(*coordination.EvictionRequest)
	oldEvictionRequest := old.(*coordination.EvictionRequest)

	allErrs = validation.ValidateEvictionRequestUpdate(evictionRequest, oldEvictionRequest)
	return rest.ValidateDeclarativelyWithMigrationChecks(ctx, legacyscheme.Scheme, evictionRequest, oldEvictionRequest, allErrs, operation.Update, rest.DeclarativeValidationConfig{})
}

func (*evictionRequestStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func (*evictionRequestStrategy) AllowUnconditionalUpdate(ctx context.Context) bool {
	return false
}

// evictionRequestStatusStrategy is the default logic invoked when updating object status.
type evictionRequestStatusStrategy struct {
	*evictionRequestStrategy
}

var _ = rest.ResetFieldsStrategy(&evictionRequestStatusStrategy{})

func NewStatusStrategy(strategy *evictionRequestStrategy) *evictionRequestStatusStrategy {
	return &evictionRequestStatusStrategy{
		strategy,
	}
}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (*evictionRequestStatusStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	return map[fieldpath.APIVersion]*fieldpath.Set{
		"coordination/v1alpha1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("spec"),
			fieldpath.MakePathOrDie("metadata"),
		),
	}
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update of status
func (*evictionRequestStatusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newEvictionRequest := obj.(*coordination.EvictionRequest)
	oldEvictionRequest := old.(*coordination.EvictionRequest)
	// Status updates should not include metadata update privileges, also,
	// the evictionrequest-controller should be responsible for the labels
	// and not the responders - let's not promote label updates.
	metav1.ResetObjectMetaForStatus(&newEvictionRequest.ObjectMeta, &oldEvictionRequest.ObjectMeta)
	newEvictionRequest.Spec = oldEvictionRequest.Spec
}

func (*evictionRequestStatusStrategy) AllowCreateOnUpdate(ctx context.Context) bool {
	return false
}

// ValidateUpdate is the default update validation for an end user updating status
func (s *evictionRequestStatusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	allErrs := validation.ValidateEvictionRequestStatusUpdate(obj.(*coordination.EvictionRequest), old.(*coordination.EvictionRequest))
	return rest.ValidateDeclarativelyWithMigrationChecks(ctx, legacyscheme.Scheme, obj, old, allErrs, operation.Update, rest.DeclarativeValidationConfig{})
}

// WarningsOnUpdate returns warnings for the given update.
func (*evictionRequestStatusStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func (*evictionRequestStatusStrategy) AllowUnconditionalUpdate(ctx context.Context) bool {
	return false
}
