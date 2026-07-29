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

package eviction

import (
	"context"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/structured-merge-diff/v6/fieldpath"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/lifecycle"
	"k8s.io/kubernetes/pkg/apis/lifecycle/validation"
	"k8s.io/utils/clock"
)

// evictionStrategy is the default logic that applies when creating and updating Eviction objects.
type evictionStrategy struct {
	rest.DeclarativeValidation
	names.NameGenerator
	clock clock.PassiveClock
}

func NewStrategy(clock clock.PassiveClock) *evictionStrategy {
	return &evictionStrategy{
		rest.DeclarativeValidation{Scheme: legacyscheme.Scheme},
		names.SimpleNameGenerator,
		clock,
	}
}

var _ = rest.ResetFieldsStrategy(&evictionStrategy{})

func (*evictionStrategy) NamespaceScoped() bool {
	return true
}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (*evictionStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"lifecycle.k8s.io/v1alpha1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
	}
	return fields
}

// PrepareForCreate clears fields that are not allowed to be set by end users on creation.
func (*evictionStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	eviction := obj.(*lifecycle.Eviction)
	eviction.Status = lifecycle.EvictionStatus{}
	eviction.Generation = 1
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (*evictionStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	oldEviction := old.(*lifecycle.Eviction)
	newEviction := obj.(*lifecycle.Eviction)
	newEviction.Status = oldEviction.Status

	// Spec updates bump the generation.
	if !apiequality.Semantic.DeepEqual(oldEviction.Spec, newEviction.Spec) {
		newEviction.Generation = oldEviction.Generation + 1
	}
}

// Validate validates a new Eviction.
func (s *evictionStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	eviction := obj.(*lifecycle.Eviction)

	allErrs := validation.ValidateEviction(eviction)
	return allErrs
}

func (*evictionStrategy) DeclarativeValidationConfig(ctx context.Context, obj, oldObj runtime.Object) rest.DeclarativeValidationConfig {
	return rest.DeclarativeValidationConfig{}
}

func (*evictionStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

func (*evictionStrategy) Canonicalize(obj runtime.Object) {
}

func (*evictionStrategy) AllowCreateOnUpdate(ctx context.Context) bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (s *evictionStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	var allErrs field.ErrorList
	eviction := obj.(*lifecycle.Eviction)
	oldEviction := old.(*lifecycle.Eviction)

	allErrs = validation.ValidateEvictionUpdate(eviction, oldEviction)
	return allErrs
}

func (*evictionStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func (*evictionStrategy) AllowUnconditionalUpdate(ctx context.Context) bool {
	return false
}

// evictionStatusStrategy is the default logic invoked when updating object status.
type evictionStatusStrategy struct {
	*evictionStrategy
}

var _ = rest.ResetFieldsStrategy(&evictionStatusStrategy{})

func NewStatusStrategy(strategy *evictionStrategy) *evictionStatusStrategy {
	return &evictionStatusStrategy{
		strategy,
	}
}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (*evictionStatusStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	return map[fieldpath.APIVersion]*fieldpath.Set{
		"lifecycle.k8s.io/v1alpha1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("spec"),
			fieldpath.MakePathOrDie("metadata"),
		),
	}
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update of status
func (*evictionStatusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newEviction := obj.(*lifecycle.Eviction)
	oldEviction := old.(*lifecycle.Eviction)
	newEviction.Spec = oldEviction.Spec
	// Status updates should not include metadata update privileges, also,
	// the eviction-controller should be responsible for the labels
	// and not the responders - let's not promote label updates.
	metav1.ResetObjectMetaForStatus(&newEviction.ObjectMeta, &oldEviction.ObjectMeta)
}

func (*evictionStatusStrategy) AllowCreateOnUpdate(ctx context.Context) bool {
	return false
}

// ValidateUpdate is the default update validation for an end user updating status
func (s *evictionStatusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	allErrs := validation.ValidateEvictionStatusUpdate(obj.(*lifecycle.Eviction), old.(*lifecycle.Eviction), validation.EvictionStatusValidationOptions{
		Clock: s.clock,
	})
	return allErrs
}

// WarningsOnUpdate returns warnings for the given update.
func (*evictionStatusStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func (*evictionStatusStrategy) AllowUnconditionalUpdate(ctx context.Context) bool {
	return false
}
