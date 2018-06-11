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

package volumesnapshot

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
	storageapi "k8s.io/kubernetes/pkg/apis/storage"
	"k8s.io/kubernetes/pkg/apis/storage/validation"
)

// volumesnapshotStrategy implements behavior for VolumeSnapshot objects
type volumesnapshotStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating VolumeSnapshot
// objects via the REST API.
var Strategy = volumesnapshotStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

func (volumesnapshotStrategy) NamespaceScoped() bool {
	return true
}

// ResetBeforeCreate clears the Status field which is not allowed to be set by end users on creation.
func (volumesnapshotStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	vs := obj.(*storageapi.VolumeSnapshot)
	vs.Status = storageapi.VolumeSnapshotStatus{}
}

func (volumesnapshotStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	volumeSnapshot := obj.(*storageapi.VolumeSnapshot)
	return validation.ValidateVolumeSnapshot(volumeSnapshot)
}

// Canonicalize normalizes the object after validation.
func (volumesnapshotStrategy) Canonicalize(obj runtime.Object) {
}

func (volumesnapshotStrategy) AllowCreateOnUpdate() bool {
	return false
}

// PrepareForUpdate sets the Status fields which is not allowed to be set by an end user updating a VS
func (volumesnapshotStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newVs := obj.(*storageapi.VolumeSnapshot)
	oldVs := old.(*storageapi.VolumeSnapshot)
	newVs.Status = oldVs.Status
}

func (volumesnapshotStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	newVs := obj.(*storageapi.VolumeSnapshot)
	errorList := validation.ValidateVolumeSnapshot(newVs)
	return append(errorList, validation.ValidateVolumeSnapshotUpdate(newVs, old.(*storageapi.VolumeSnapshot))...)
}

func (volumesnapshotStrategy) AllowUnconditionalUpdate() bool {
	return true
}

type volumeSnapshotStatusStrategy struct {
	volumesnapshotStrategy
}

var StatusStrategy = volumeSnapshotStatusStrategy{Strategy}

// PrepareForUpdate sets the Spec field which is not allowed to be changed when updating a VS's Status
func (volumeSnapshotStatusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newVs := obj.(*storageapi.VolumeSnapshot)
	oldVs := old.(*storageapi.VolumeSnapshot)
	newVs.Spec = oldVs.Spec
}

func (volumeSnapshotStatusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateVolumeSnapshotStatusUpdate(obj.(*storageapi.VolumeSnapshot), old.(*storageapi.VolumeSnapshot))
}

// GetAttrs returns labels and fields of a given object for filtering purposes.
func GetAttrs(obj runtime.Object) (labels.Set, fields.Set, bool, error) {
	volumeSnapshotObj, ok := obj.(*storageapi.VolumeSnapshot)
	if !ok {
		return nil, nil, false, fmt.Errorf("not a volumeSnapshot")
	}
	return labels.Set(volumeSnapshotObj.Labels), VolumeSnapshotToSelectableFields(volumeSnapshotObj), volumeSnapshotObj.Initializers != nil, nil
}

// MatchVolumeSnapshots returns a generic matcher for a given label and field selector.
func MatchVolumeSnapshots(label labels.Selector, field fields.Selector) storage.SelectionPredicate {
	return storage.SelectionPredicate{
		Label:    label,
		Field:    field,
		GetAttrs: GetAttrs,
	}
}

// VolumeSnapshotToSelectableFields returns a field set that represents the object
func VolumeSnapshotToSelectableFields(volumeSnapshot *storageapi.VolumeSnapshot) fields.Set {
	objectMetaFieldsSet := generic.ObjectMetaFieldsSet(&volumeSnapshot.ObjectMeta, true)
	specificFieldsSet := fields.Set{
		// This is a bug, but we need to support it for backward compatibility.
		"name": volumeSnapshot.Name,
	}
	return generic.MergeFieldsSets(objectMetaFieldsSet, specificFieldsSet)
}
