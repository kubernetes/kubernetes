/*
Copyright 2020 The Kubernetes Authors.

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

package csistoragecapacity

import (
	"context"

	metav1validation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	storageutil "k8s.io/kubernetes/pkg/api/storage"
	"k8s.io/kubernetes/pkg/apis/storage"
	"k8s.io/kubernetes/pkg/apis/storage/validation"
)

// csiStorageCapacityStrategy implements behavior for CSIStorageCapacity objects
type csiStorageCapacityStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating
// CSIStorageCapacity objects via the REST API.
var Strategy = csiStorageCapacityStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

func (csiStorageCapacityStrategy) NamespaceScoped() bool {
	return true
}

// PrepareForCreate is currently a NOP.
func (csiStorageCapacityStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
}

func (csiStorageCapacityStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	csiStorageCapacity := obj.(*storage.CSIStorageCapacity)
	opts := validation.CSIStorageCapacityValidateOptions{
		AllowInvalidLabelValueInSelector: false,
	}
	errs := validation.ValidateCSIStorageCapacity(csiStorageCapacity, opts)
	return errs
}

// WarningsOnCreate returns warnings for the creation of the given object.
func (csiStorageCapacityStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return storageutil.GetWarningsForCSIStorageCapacity(obj.(*storage.CSIStorageCapacity))
}

// Canonicalize normalizes the object after validation.
func (csiStorageCapacityStrategy) Canonicalize(obj runtime.Object) {
}

func (csiStorageCapacityStrategy) AllowCreateOnUpdate() bool {
	return false
}

// PrepareForUpdate is currently a NOP.
func (csiStorageCapacityStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
}

func (csiStorageCapacityStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	newCSIStorageCapacityObj := obj.(*storage.CSIStorageCapacity)
	oldCSIStorageCapacityObj := old.(*storage.CSIStorageCapacity)
	opts := validation.CSIStorageCapacityValidateOptions{
		AllowInvalidLabelValueInSelector: hasInvalidLabelValueInLabelSelector(oldCSIStorageCapacityObj),
	}
	errorList := validation.ValidateCSIStorageCapacity(newCSIStorageCapacityObj, opts)
	return append(errorList, validation.ValidateCSIStorageCapacityUpdate(newCSIStorageCapacityObj, oldCSIStorageCapacityObj)...)
}

// WarningsOnUpdate returns warnings for the given update.
func (csiStorageCapacityStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return storageutil.GetWarningsForCSIStorageCapacity(obj.(*storage.CSIStorageCapacity))
}

func (csiStorageCapacityStrategy) AllowUnconditionalUpdate() bool {
	return false
}

func hasInvalidLabelValueInLabelSelector(capacity *storage.CSIStorageCapacity) bool {
	labelSelectorValidationOptions := metav1validation.LabelSelectorValidationOptions{AllowInvalidLabelValueInSelector: false}
	return len(metav1validation.ValidateLabelSelector(capacity.NodeTopology, labelSelectorValidationOptions, nil)) > 0
}
