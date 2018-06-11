/*
Copyright 2018 The Kubernetes Authors.

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

package volumesnapshotdata

import (
	"context"
	"fmt"

	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	apistorage "k8s.io/kubernetes/pkg/apis/storage"
	"k8s.io/kubernetes/pkg/apis/storage/validation"
)

// volumesnapshotdataStrategy implements behavior for VolumeSnapshotData objects
type volumesnapshotdataStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating VolumeSnapshotData
// objects via the REST API.
var Strategy = volumesnapshotdataStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

func (volumesnapshotdataStrategy) NamespaceScoped() bool {
	return false
}

// ResetBeforeCreate clears the Status field which is not allowed to be set by end users on creation.
func (volumesnapshotdataStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
}

func (volumesnapshotdataStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	volumeSnapshotData := obj.(*apistorage.VolumeSnapshotData)
	return validation.ValidateVolumeSnapshotData(volumeSnapshotData)
}

// Canonicalize normalizes the object after validation.
func (volumesnapshotdataStrategy) Canonicalize(obj runtime.Object) {
}

func (volumesnapshotdataStrategy) AllowCreateOnUpdate() bool {
	return false
}

// PrepareForUpdate sets the Status fields which is not allowed to be set by an end user updating a VSD
func (volumesnapshotdataStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newVsd := obj.(*apistorage.VolumeSnapshotData)
	oldVsd := old.(*apistorage.VolumeSnapshotData)
	newVsd.Status = oldVsd.Status
}

func (volumesnapshotdataStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	newVsd := obj.(*apistorage.VolumeSnapshotData)
	errorList := validation.ValidateVolumeSnapshotData(newVsd)
	return append(errorList, validation.ValidateVolumeSnapshotDataUpdate(newVsd, old.(*apistorage.VolumeSnapshotData))...)
}

func (volumesnapshotdataStrategy) AllowUnconditionalUpdate() bool {
	return true
}

type volumeSnapshotStatusStrategy struct {
	volumesnapshotdataStrategy
}

var StatusStrategy = volumeSnapshotStatusStrategy{Strategy}

// PrepareForUpdate sets the Spec field which is not allowed to be changed when updating a PV's Status
func (volumeSnapshotStatusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newVsd := obj.(*apistorage.VolumeSnapshotData)
	oldVsd := old.(*apistorage.VolumeSnapshotData)
	newVsd.Spec = oldVsd.Spec
}

func (volumeSnapshotStatusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateVolumeSnapshotDataStatusUpdate(obj.(*apistorage.VolumeSnapshotData), old.(*apistorage.VolumeSnapshotData))
}

// GetAttrs returns labels and fields of a given object for filtering purposes.
func GetAttrs(obj runtime.Object) (labels.Set, fields.Set, bool, error) {
	volumeSnapshotDataObj, ok := obj.(*apistorage.VolumeSnapshotData)
	if !ok {
		return nil, nil, false, fmt.Errorf("not a volumesnapshotdata")
	}
	return labels.Set(volumeSnapshotDataObj.Labels), VolumeSnapshotDataToSelectableFields(volumeSnapshotDataObj), volumeSnapshotDataObj.Initializers != nil, nil
}

// MatchVolumeSnapshotData returns a generic matcher for a given label and field selector.
func MatchVolumeSnapshotData(label labels.Selector, field fields.Selector) storage.SelectionPredicate {
	return storage.SelectionPredicate{
		Label:    label,
		Field:    field,
		GetAttrs: GetAttrs,
	}
}

// VolumeSnapshotDataToSelectableFields returns a field set that represents the object
func VolumeSnapshotDataToSelectableFields(VolumeSnapshotData *apistorage.VolumeSnapshotData) fields.Set {
	objectMetaFieldsSet := generic.ObjectMetaFieldsSet(&VolumeSnapshotData.ObjectMeta, false)
	specificFieldsSet := fields.Set{
		// This is a bug, but we need to support it for backward compatibility.
		"name": VolumeSnapshotData.Name,
	}
	return generic.MergeFieldsSets(objectMetaFieldsSet, specificFieldsSet)
}
