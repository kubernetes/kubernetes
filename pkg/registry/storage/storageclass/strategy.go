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

package storageclass

import (
	"context"
	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apiserver/pkg/registry/rest"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	storageutil "k8s.io/kubernetes/pkg/api/storage"
	"k8s.io/kubernetes/pkg/apis/storage"
	"k8s.io/kubernetes/pkg/apis/storage/validation"
)

// storageClassStrategy implements behavior for StorageClass objects
type storageClassStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating
// StorageClass objects via the REST API.
var Strategy = storageClassStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

func (storageClassStrategy) NamespaceScoped() bool {
	return false
}

// ResetBeforeCreate clears the Status field which is not allowed to be set by end users on creation.
func (storageClassStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
}

func (storageClassStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	storageClass := obj.(*storage.StorageClass)
	allErrs := validation.ValidateStorageClass(storageClass)
	return rest.ValidateDeclarativelyWithMigrationChecks(ctx, legacyscheme.Scheme, obj, nil, allErrs, operation.Create)
}

// WarningsOnCreate returns warnings for the creation of the given object.
func (storageClassStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return storageutil.GetWarningsForStorageClass(obj.(*storage.StorageClass))
}

// Canonicalize normalizes the object after validation.
func (storageClassStrategy) Canonicalize(obj runtime.Object) {
}

func (storageClassStrategy) AllowCreateOnUpdate() bool {
	return false
}

// PrepareForUpdate sets the Status fields which is not allowed to be set by an end user updating a PV
func (storageClassStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
}

func (storageClassStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	allErrs := validation.ValidateStorageClass(obj.(*storage.StorageClass))
	allErrs = append(allErrs, validation.ValidateStorageClassUpdate(obj.(*storage.StorageClass), old.(*storage.StorageClass))...)
	return rest.ValidateDeclarativelyWithMigrationChecks(ctx, legacyscheme.Scheme, obj, old, allErrs, operation.Update)
}

// WarningsOnUpdate returns warnings for the given update.
func (storageClassStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return storageutil.GetWarningsForStorageClass(obj.(*storage.StorageClass))
}

func (storageClassStrategy) AllowUnconditionalUpdate() bool {
	return true
}
