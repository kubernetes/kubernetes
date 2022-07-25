/*
Copyright 2016 The Kubernetes Authors.

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

package apiservice

import (
	"context"
	"fmt"

	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/names"

	"k8s.io/kube-aggregator/pkg/apis/apiregistration"
	"k8s.io/kube-aggregator/pkg/apis/apiregistration/validation"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
)

type apiServerStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// apiServerStrategy must implement rest.RESTCreateUpdateStrategy
var _ rest.RESTCreateUpdateStrategy = apiServerStrategy{}
var Strategy = apiServerStrategy{}

// NewStrategy creates a new apiServerStrategy.
func NewStrategy(typer runtime.ObjectTyper) rest.CreateUpdateResetFieldsStrategy {
	return apiServerStrategy{typer, names.SimpleNameGenerator}
}

func (apiServerStrategy) NamespaceScoped() bool {
	return false
}

func (apiServerStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"apiregistration.k8s.io/v1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
		"apiregistration.k8s.io/v1beta1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
	}

	return fields
}

func (apiServerStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	apiservice := obj.(*apiregistration.APIService)
	apiservice.Status = apiregistration.APIServiceStatus{}

	// mark local API services as immediately available on create
	if apiservice.Spec.Service == nil {
		apiregistration.SetAPIServiceCondition(apiservice, apiregistration.NewLocalAvailableAPIServiceCondition())
	}
}

func (apiServerStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newAPIService := obj.(*apiregistration.APIService)
	oldAPIService := old.(*apiregistration.APIService)
	newAPIService.Status = oldAPIService.Status
}

func (apiServerStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	return validation.ValidateAPIService(obj.(*apiregistration.APIService))
}

// WarningsOnCreate returns warnings for the creation of the given object.
func (apiServerStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

func (apiServerStrategy) AllowCreateOnUpdate() bool {
	return false
}

func (apiServerStrategy) AllowUnconditionalUpdate() bool {
	return false
}

func (apiServerStrategy) Canonicalize(obj runtime.Object) {
}

func (apiServerStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateAPIServiceUpdate(obj.(*apiregistration.APIService), old.(*apiregistration.APIService))
}

// WarningsOnUpdate returns warnings for the given update.
func (apiServerStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

type apiServerStatusStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// NewStatusStrategy creates a new apiServerStatusStrategy.
func NewStatusStrategy(typer runtime.ObjectTyper) rest.UpdateResetFieldsStrategy {
	return apiServerStatusStrategy{typer, names.SimpleNameGenerator}
}

func (apiServerStatusStrategy) NamespaceScoped() bool {
	return false
}

func (apiServerStatusStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"apiregistration.k8s.io/v1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("spec"),
			fieldpath.MakePathOrDie("metadata"),
		),
		"apiregistration.k8s.io/v1beta1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("spec"),
			fieldpath.MakePathOrDie("metadata"),
		),
	}

	return fields
}

func (apiServerStatusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newAPIService := obj.(*apiregistration.APIService)
	oldAPIService := old.(*apiregistration.APIService)
	newAPIService.Spec = oldAPIService.Spec
	newAPIService.Labels = oldAPIService.Labels
	newAPIService.Annotations = oldAPIService.Annotations
	newAPIService.Finalizers = oldAPIService.Finalizers
	newAPIService.OwnerReferences = oldAPIService.OwnerReferences
}

func (apiServerStatusStrategy) AllowCreateOnUpdate() bool {
	return false
}

func (apiServerStatusStrategy) AllowUnconditionalUpdate() bool {
	return false
}

// Canonicalize normalizes the object after validation.
func (apiServerStatusStrategy) Canonicalize(obj runtime.Object) {
}

// ValidateUpdate validates an update of apiServerStatusStrategy.
func (apiServerStatusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateAPIServiceStatusUpdate(obj.(*apiregistration.APIService), old.(*apiregistration.APIService))
}

// WarningsOnUpdate returns warnings for the given update.
func (apiServerStatusStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

// GetAttrs returns the labels and fields of an API server for filtering purposes.
func GetAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	apiserver, ok := obj.(*apiregistration.APIService)
	if !ok {
		return nil, nil, fmt.Errorf("given object is not a APIService")
	}
	return labels.Set(apiserver.ObjectMeta.Labels), ToSelectableFields(apiserver), nil
}

// MatchAPIService is the filter used by the generic etcd backend to watch events
// from etcd to clients of the apiserver only interested in specific labels/fields.
func MatchAPIService(label labels.Selector, field fields.Selector) storage.SelectionPredicate {
	return storage.SelectionPredicate{
		Label:    label,
		Field:    field,
		GetAttrs: GetAttrs,
	}
}

// ToSelectableFields returns a field set that represents the object.
func ToSelectableFields(obj *apiregistration.APIService) fields.Set {
	return generic.ObjectMetaFieldsSet(&obj.ObjectMeta, true)
}
