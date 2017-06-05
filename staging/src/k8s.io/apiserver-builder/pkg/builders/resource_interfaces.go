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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/names"
)

type HasObjectMetaSpecStatus interface {
	HasObjectMeta
	HasSpec
	HasStatus
}

type HasStatus interface {
	NewStatus() interface{}
	GetStatus() interface{}
	SetStatus(status interface{})
}

type HasSpec interface {
	GetSpec() interface{}
	SetSpec(spec interface{})
}

type HasObjectMeta interface {
	GetObjectMeta() *metav1.ObjectMeta
}

type StorageBuilder interface {
	Build(builder StorageBuilder, store *StorageWrapper, options *generic.StoreOptions)

	names.NameGenerator
	runtime.ObjectTyper

	ObjectNameFunc(obj runtime.Object) (string, error)
	NamespaceScoped() bool
	AllowCreateOnUpdate() bool
	AllowUnconditionalUpdate() bool
	Canonicalize(obj runtime.Object)
	PrepareForCreate(ctx request.Context, obj runtime.Object)
	PrepareForUpdate(ctx request.Context, obj, old runtime.Object)
	Validate(ctx request.Context, obj runtime.Object) field.ErrorList
	ValidateUpdate(ctx request.Context, obj, old runtime.Object) field.ErrorList
	GetAttrs(obj runtime.Object) (labels.Set, fields.Set, error)
	GetSelectableFields(obj HasObjectMeta) fields.Set
	BasicMatch(label labels.Selector, field fields.Selector) storage.SelectionPredicate
}

type SchemeFns interface {
	GetDefaultingFunctions() []interface{}
	GetConversionFunctions() []interface{}
	Register(scheme *runtime.Scheme) error
}

type StandardStorageProvider interface {
	GetStandardStorage() rest.StandardStorage
}
