/*
Copyright 2022 The Kubernetes Authors.

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

package carp

import (
	"context"
	"errors"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/testapigroup"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/names"
	v1 "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"sigs.k8s.io/structured-merge-diff/v6/fieldpath"
)

// carpStrategy implements behavior for Carp objects
type carpStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
	nsClient v1.NamespaceInterface
}

// NewStrategy is the default logic that applies when creating and updating Carp objects.
func NewStrategy(nsClient v1.NamespaceInterface) *carpStrategy {
	return &carpStrategy{
		legacyscheme.Scheme,
		names.SimpleNameGenerator,
		nsClient,
	}
}

func (*carpStrategy) NamespaceScoped() bool {
	return true
}

// GetResetFields returns the set of fields that get reset by the strategy and
// should not be modified by the user. For a new Carp that is the
// status.
func (*carpStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"testapigroup.apimachinery.k8s.io/v1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
	}

	return fields
}

func (*carpStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	claim := obj.(*testapigroup.Carp)
	// Status must not be set by user on create.
	claim.Status = testapigroup.CarpStatus{}
}

func (s *carpStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	return nil
}

func (*carpStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

func (*carpStrategy) Canonicalize(obj runtime.Object) {
}

func (*carpStrategy) AllowCreateOnUpdate() bool {
	return false
}

func (*carpStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newClaim := obj.(*testapigroup.Carp)
	oldClaim := old.(*testapigroup.Carp)
	newClaim.Status = oldClaim.Status
}

func (s *carpStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return nil
}

func (*carpStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func (*carpStrategy) AllowUnconditionalUpdate() bool {
	return true
}

type carpStatusStrategy struct {
	*carpStrategy
}

// NewStatusStrategy creates a strategy for operating the status object.
func NewStatusStrategy(carpStrategy *carpStrategy) *carpStatusStrategy {
	return &carpStatusStrategy{carpStrategy}
}

// GetResetFields returns the set of fields that get reset by the strategy and
// should not be modified by the user. For a status update that is the spec.
func (*carpStatusStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"testapigroup.apimachinery.k8s.io/v1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("spec"),
		),
	}

	return fields
}

func (*carpStatusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newClaim := obj.(*testapigroup.Carp)
	oldClaim := old.(*testapigroup.Carp)
	newClaim.Spec = oldClaim.Spec
	metav1.ResetObjectMetaForStatus(&newClaim.ObjectMeta, &oldClaim.ObjectMeta)
}

func (r *carpStatusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return nil
}

// WarningsOnUpdate returns warnings for the given update.
func (*carpStatusStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

// Match returns a generic matcher for a given label and field selector.
func Match(label labels.Selector, field fields.Selector) storage.SelectionPredicate {
	return storage.SelectionPredicate{
		Label:    label,
		Field:    field,
		GetAttrs: GetAttrs,
	}
}

// GetAttrs returns labels and fields of a given object for filtering purposes.
func GetAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	claim, ok := obj.(*testapigroup.Carp)
	if !ok {
		return nil, nil, errors.New("not a carp")
	}
	return labels.Set(claim.Labels), toSelectableFields(claim), nil
}

// toSelectableFields returns a field set that represents the object
func toSelectableFields(claim *testapigroup.Carp) fields.Set {
	fields := generic.ObjectMetaFieldsSet(&claim.ObjectMeta, true)
	return fields
}
