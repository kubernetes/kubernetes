/*
Copyright 2023 The Kubernetes Authors.

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

package volumeattributesclass

import (
	"context"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/storage"
	"k8s.io/kubernetes/pkg/apis/storage/validation"
)

// volumeAttributesClassStrategy implements behavior for VolumeAttributesClassStrategy objects
type volumeAttributesClassStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating
// VolumeAttributesClass objects via the REST API.
var Strategy = volumeAttributesClassStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

func (volumeAttributesClassStrategy) NamespaceScoped() bool {
	return false
}

// ResetBeforeCreate clears the Status field which is not allowed to be set by end users on creation.
func (volumeAttributesClassStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
}

func (volumeAttributesClassStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	volumeAttributesClass := obj.(*storage.VolumeAttributesClass)
	return validation.ValidateVolumeAttributesClass(volumeAttributesClass)
}

// WarningsOnCreate returns warnings for the creation of the given object.
func (volumeAttributesClassStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

// Canonicalize normalizes the object after validation.
func (volumeAttributesClassStrategy) Canonicalize(obj runtime.Object) {
}

func (volumeAttributesClassStrategy) AllowCreateOnUpdate() bool {
	return false
}

// PrepareForUpdate sets the Status fields which is not allowed to be set by an end user updating a PV
func (volumeAttributesClassStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
}

func (volumeAttributesClassStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	errorList := validation.ValidateVolumeAttributesClass(obj.(*storage.VolumeAttributesClass))
	return append(errorList, validation.ValidateVolumeAttributesClassUpdate(obj.(*storage.VolumeAttributesClass), old.(*storage.VolumeAttributesClass))...)
}

// WarningsOnUpdate returns warnings for the given update.
func (volumeAttributesClassStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func (volumeAttributesClassStrategy) AllowUnconditionalUpdate() bool {
	return true
}
