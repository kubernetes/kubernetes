/*
Copyright 2019 The Kubernetes Authors.

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

package csidriver

import (
	"context"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/storage/names"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/storage"
	"k8s.io/kubernetes/pkg/apis/storage/validation"
	"k8s.io/kubernetes/pkg/features"
)

// csiDriverStrategy implements behavior for CSIDriver objects
type csiDriverStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating
// CSIDriver objects via the REST API.
var Strategy = csiDriverStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

func (csiDriverStrategy) NamespaceScoped() bool {
	return false
}

// PrepareForCreate clears the VolumeLifecycleModes field if the corresponding feature is disabled.
func (csiDriverStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.CSIInlineVolume) {
		csiDriver := obj.(*storage.CSIDriver)
		csiDriver.Spec.VolumeLifecycleModes = nil
	}
}

func (csiDriverStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	csiDriver := obj.(*storage.CSIDriver)

	errs := validation.ValidateCSIDriver(csiDriver)
	errs = append(errs, validation.ValidateCSIDriver(csiDriver)...)

	return errs
}

// Canonicalize normalizes the object after validation.
func (csiDriverStrategy) Canonicalize(obj runtime.Object) {
}

func (csiDriverStrategy) AllowCreateOnUpdate() bool {
	return false
}

// PrepareForUpdate clears the VolumeLifecycleModes field if the corresponding feature is disabled and
// existing object does not already have that field set. This allows the field to remain when
// downgrading to a version that has the feature disabled.
func (csiDriverStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	if old.(*storage.CSIDriver).Spec.VolumeLifecycleModes == nil &&
		!utilfeature.DefaultFeatureGate.Enabled(features.CSIInlineVolume) {
		newCSIDriver := obj.(*storage.CSIDriver)
		newCSIDriver.Spec.VolumeLifecycleModes = nil
	}
}

func (csiDriverStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	newCSIDriverObj := obj.(*storage.CSIDriver)
	oldCSIDriverObj := old.(*storage.CSIDriver)
	errorList := validation.ValidateCSIDriver(newCSIDriverObj)
	return append(errorList, validation.ValidateCSIDriverUpdate(newCSIDriverObj, oldCSIDriverObj)...)
}

func (csiDriverStrategy) AllowUnconditionalUpdate() bool {
	return false
}
