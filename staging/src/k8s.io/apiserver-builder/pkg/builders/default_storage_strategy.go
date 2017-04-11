/*
Copyright 2017 The Kubernetes Authors.

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

package builders

import (
	"fmt"
	"reflect"

	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/client-go/pkg/api"
)

var _ rest.RESTCreateStrategy = &DefaultStorageStrategy{}
var _ rest.RESTDeleteStrategy = &DefaultStorageStrategy{}
var _ rest.RESTUpdateStrategy = &DefaultStorageStrategy{}

var StorageStrategySingleton = DefaultStorageStrategy{
	api.Scheme,
	names.SimpleNameGenerator,
}

type DefaultStorageStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

func (DefaultStorageStrategy) ObjectNameFunc(obj runtime.Object) (string, error) {
	switch obj := obj.(type) {
	default:
		return "", fmt.Errorf(
			"Cannot get name for object type %T.  Must implement HasObjectMeta or define "+
				"its own ObjectNameFunc in its storage strategy.", obj)
	case HasObjectMeta:
		// Get the name from the metadata
		return obj.GetObjectMeta().Name, nil
	}
}

// Build sets the strategy for the store
func (DefaultStorageStrategy) Build(builder StorageBuilder, store *StorageWrapper, options *generic.StoreOptions) {
	store.PredicateFunc = builder.BasicMatch
	store.ObjectNameFunc = builder.ObjectNameFunc
	store.CreateStrategy = builder
	store.UpdateStrategy = builder
	store.DeleteStrategy = builder

	options.AttrFunc = builder.GetAttrs
}

func (DefaultStorageStrategy) NamespaceScoped() bool { return true }

func (DefaultStorageStrategy) AllowCreateOnUpdate() bool { return true }

func (DefaultStorageStrategy) AllowUnconditionalUpdate() bool { return true }

func (DefaultStorageStrategy) Canonicalize(obj runtime.Object) {}

func (DefaultStorageStrategy) PrepareForCreate(ctx request.Context, obj runtime.Object) {
	switch t := obj.(type) {
	default:
	case HasObjectMetaSpecStatus:
		// Clear the status if the resource has a Status
		t.GetObjectMeta().Generation = 1
		t.SetStatus(t.NewStatus())
	}
}

func (DefaultStorageStrategy) PrepareForUpdate(ctx request.Context, obj, old runtime.Object) {
	// Don't update the status if the resource has a Status
	switch n := obj.(type) {
	default:
	case HasObjectMetaSpecStatus:
		o := old.(HasObjectMetaSpecStatus)
		n.SetStatus(o.GetStatus())

		// Spec and annotation updates bump the generation.
		if !reflect.DeepEqual(n.GetSpec(), o.GetSpec()) ||
			!reflect.DeepEqual(n.GetObjectMeta().Annotations, o.GetObjectMeta().Annotations) {
			n.GetObjectMeta().Generation = o.GetObjectMeta().Generation + 1
		}
	}
}

func (DefaultStorageStrategy) Validate(ctx request.Context, obj runtime.Object) field.ErrorList {
	return field.ErrorList{}
}

func (DefaultStorageStrategy) ValidateUpdate(ctx request.Context, obj, old runtime.Object) field.ErrorList {
	return field.ErrorList{}
}

func (b DefaultStorageStrategy) GetAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	switch t := obj.(type) {
	case HasObjectMeta:
		apiserver := obj.(HasObjectMeta)
		return labels.Set(apiserver.GetObjectMeta().Labels), b.GetSelectableFields(apiserver), nil
	default:
		return nil, nil, fmt.Errorf(
			"Cannot get attributes for object type %v which does not implement HasObjectMeta.", t)
	}
}

// GetSelectableFields returns a field set that represents the object.
func (DefaultStorageStrategy) GetSelectableFields(obj HasObjectMeta) fields.Set {
	return generic.ObjectMetaFieldsSet(obj.GetObjectMeta(), true)
}

// MatchResource is the filter used by the generic etcd backend to watch events
// from etcd to clients of the apiserver only interested in specific labels/fields.
func (b DefaultStorageStrategy) BasicMatch(label labels.Selector, field fields.Selector) storage.SelectionPredicate {
	return storage.SelectionPredicate{
		Label:    label,
		Field:    field,
		GetAttrs: b.GetAttrs,
	}
}

//
// Status Strategies
//

var StatusStorageStrategySingleton = DefaultStatusStorageStrategy{StorageStrategySingleton}

type DefaultStatusStorageStrategy struct {
	DefaultStorageStrategy
}

func (DefaultStatusStorageStrategy) PrepareForUpdate(ctx request.Context, obj, old runtime.Object) {
	switch n := obj.(type) {
	default:
	case HasObjectMetaSpecStatus:
		// Only update the Status
		o := old.(HasObjectMetaSpecStatus)
		n.SetSpec(o.GetSpec())
		n.GetObjectMeta().Labels = o.GetObjectMeta().Labels
	}
}
