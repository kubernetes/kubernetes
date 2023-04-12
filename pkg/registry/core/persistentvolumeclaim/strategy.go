/*
Copyright 2014 The Kubernetes Authors.

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

package persistentvolumeclaim

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
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"

	"k8s.io/kubernetes/pkg/api/legacyscheme"
	pvcutil "k8s.io/kubernetes/pkg/api/persistentvolumeclaim"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/validation"
)

// persistentvolumeclaimStrategy implements behavior for PersistentVolumeClaim objects
type persistentvolumeclaimStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating PersistentVolumeClaim
// objects via the REST API.
var Strategy = persistentvolumeclaimStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

func (persistentvolumeclaimStrategy) NamespaceScoped() bool {
	return true
}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (persistentvolumeclaimStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"v1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
	}

	return fields
}

// PrepareForCreate clears the Status field which is not allowed to be set by end users on creation.
func (persistentvolumeclaimStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	pvc := obj.(*api.PersistentVolumeClaim)
	pvc.Status = api.PersistentVolumeClaimStatus{}
	pvcutil.DropDisabledFields(&pvc.Spec, nil)

	// For data sources, we need to do 2 things to implement KEP 1495

	// First drop invalid values from spec.dataSource (anything other than PVC or
	// VolumeSnapshot) if certain conditions are met.
	pvcutil.EnforceDataSourceBackwardsCompatibility(&pvc.Spec, nil)

	// Second copy dataSource -> dataSourceRef or dataSourceRef -> dataSource if one of them
	// is nil and the other is non-nil
	pvcutil.NormalizeDataSources(&pvc.Spec)
}

func (persistentvolumeclaimStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	pvc := obj.(*api.PersistentVolumeClaim)
	opts := validation.ValidationOptionsForPersistentVolumeClaim(pvc, nil)
	return validation.ValidatePersistentVolumeClaim(pvc, opts)
}

// WarningsOnCreate returns warnings for the creation of the given object.
func (persistentvolumeclaimStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return pvcutil.GetWarningsForPersistentVolumeClaim(obj.(*api.PersistentVolumeClaim))
}

// Canonicalize normalizes the object after validation.
func (persistentvolumeclaimStrategy) Canonicalize(obj runtime.Object) {
}

func (persistentvolumeclaimStrategy) AllowCreateOnUpdate() bool {
	return false
}

// PrepareForUpdate sets the Status field which is not allowed to be set by end users on update
func (persistentvolumeclaimStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newPvc := obj.(*api.PersistentVolumeClaim)
	oldPvc := old.(*api.PersistentVolumeClaim)
	newPvc.Status = oldPvc.Status

	pvcutil.DropDisabledFields(&newPvc.Spec, &oldPvc.Spec)

	// We need to use similar logic to PrepareForCreate here both to preserve backwards
	// compatibility with the old behavior (ignoring of garbage dataSources at both create
	// and update time) and also for compatibility with older clients, that might omit
	// the dataSourceRef field which we filled in automatically, so we have to fill it
	// in again here.
	pvcutil.EnforceDataSourceBackwardsCompatibility(&newPvc.Spec, &oldPvc.Spec)
	pvcutil.NormalizeDataSources(&newPvc.Spec)

	// We also normalize the data source fields of the old PVC, so that objects saved
	// from an earlier version will pass validation.
	pvcutil.NormalizeDataSources(&oldPvc.Spec)
}

func (persistentvolumeclaimStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	newPvc := obj.(*api.PersistentVolumeClaim)
	oldPvc := old.(*api.PersistentVolumeClaim)
	opts := validation.ValidationOptionsForPersistentVolumeClaim(newPvc, oldPvc)
	errorList := validation.ValidatePersistentVolumeClaim(newPvc, opts)
	return append(errorList, validation.ValidatePersistentVolumeClaimUpdate(newPvc, oldPvc, opts)...)
}

// WarningsOnUpdate returns warnings for the given update.
func (persistentvolumeclaimStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return pvcutil.GetWarningsForPersistentVolumeClaim(obj.(*api.PersistentVolumeClaim))
}

func (persistentvolumeclaimStrategy) AllowUnconditionalUpdate() bool {
	return true
}

type persistentvolumeclaimStatusStrategy struct {
	persistentvolumeclaimStrategy
}

var StatusStrategy = persistentvolumeclaimStatusStrategy{Strategy}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (persistentvolumeclaimStatusStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"v1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("spec"),
		),
	}

	return fields
}

// PrepareForUpdate sets the Spec field which is not allowed to be changed when updating a PV's Status
func (persistentvolumeclaimStatusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newPVC := obj.(*api.PersistentVolumeClaim)
	oldPVC := old.(*api.PersistentVolumeClaim)
	newPVC.Spec = oldPVC.Spec
	pvcutil.DropDisabledFieldsFromStatus(newPVC, oldPVC)
}

func (persistentvolumeclaimStatusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	newPvc := obj.(*api.PersistentVolumeClaim)
	oldPvc := old.(*api.PersistentVolumeClaim)
	opts := validation.ValidationOptionsForPersistentVolumeClaim(newPvc, oldPvc)
	return validation.ValidatePersistentVolumeClaimStatusUpdate(newPvc, oldPvc, opts)
}

// WarningsOnUpdate returns warnings for the given update.
func (persistentvolumeclaimStatusStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

// GetAttrs returns labels and fields of a given object for filtering purposes.
func GetAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	persistentvolumeclaimObj, ok := obj.(*api.PersistentVolumeClaim)
	if !ok {
		return nil, nil, fmt.Errorf("not a persistentvolumeclaim")
	}
	return labels.Set(persistentvolumeclaimObj.Labels), PersistentVolumeClaimToSelectableFields(persistentvolumeclaimObj), nil
}

// MatchPersistentVolumeClaim returns a generic matcher for a given label and field selector.
func MatchPersistentVolumeClaim(label labels.Selector, field fields.Selector) storage.SelectionPredicate {
	return storage.SelectionPredicate{
		Label:    label,
		Field:    field,
		GetAttrs: GetAttrs,
	}
}

// PersistentVolumeClaimToSelectableFields returns a field set that represents the object
func PersistentVolumeClaimToSelectableFields(persistentvolumeclaim *api.PersistentVolumeClaim) fields.Set {
	objectMetaFieldsSet := generic.ObjectMetaFieldsSet(&persistentvolumeclaim.ObjectMeta, true)
	specificFieldsSet := fields.Set{
		// This is a bug, but we need to support it for backward compatibility.
		"name": persistentvolumeclaim.Name,
	}
	return generic.MergeFieldsSets(objectMetaFieldsSet, specificFieldsSet)
}
