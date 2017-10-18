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
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/storage/names"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/storage"
	"k8s.io/kubernetes/pkg/apis/storage/validation"
	"k8s.io/kubernetes/pkg/features"
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
func (storageClassStrategy) PrepareForCreate(ctx genericapirequest.Context, obj runtime.Object) {
	class := obj.(*storage.StorageClass)

	if !utilfeature.DefaultFeatureGate.Enabled(features.ExpandPersistentVolumes) {
		class.AllowVolumeExpansion = nil
	}
}

func (storageClassStrategy) Validate(ctx genericapirequest.Context, obj runtime.Object) field.ErrorList {
	storageClass := obj.(*storage.StorageClass)
	return validation.ValidateStorageClass(storageClass)
}

// Canonicalize normalizes the object after validation.
func (storageClassStrategy) Canonicalize(obj runtime.Object) {
}

func (storageClassStrategy) AllowCreateOnUpdate() bool {
	return false
}

// PrepareForUpdate sets the Status fields which is not allowed to be set by an end user updating a PV
func (storageClassStrategy) PrepareForUpdate(ctx genericapirequest.Context, obj, old runtime.Object) {
	newClass := obj.(*storage.StorageClass)
	oldClass := old.(*storage.StorageClass)

	if !utilfeature.DefaultFeatureGate.Enabled(features.ExpandPersistentVolumes) {
		newClass.AllowVolumeExpansion = nil
		oldClass.AllowVolumeExpansion = nil
	}
}

func (storageClassStrategy) ValidateUpdate(ctx genericapirequest.Context, obj, old runtime.Object) field.ErrorList {
	errorList := validation.ValidateStorageClass(obj.(*storage.StorageClass))
	return append(errorList, validation.ValidateStorageClassUpdate(obj.(*storage.StorageClass), old.(*storage.StorageClass))...)
}

func (storageClassStrategy) AllowUnconditionalUpdate() bool {
	return true
}
