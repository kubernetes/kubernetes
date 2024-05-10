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

package persistentvolume

import (
	"context"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/features"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	pvutil "k8s.io/kubernetes/pkg/api/persistentvolume"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/validation"
	volumevalidation "k8s.io/kubernetes/pkg/volume/validation"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
)

// persistentvolumeStrategy implements behavior for PersistentVolume objects
type persistentvolumeStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating PersistentVolume
// objects via the REST API.
var Strategy = persistentvolumeStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

func (persistentvolumeStrategy) NamespaceScoped() bool {
	return false
}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (persistentvolumeStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"v1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
	}

	return fields
}

// ResetBeforeCreate clears the Status field which is not allowed to be set by end users on creation.
func (persistentvolumeStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	pv := obj.(*api.PersistentVolume)
	pv.Status = api.PersistentVolumeStatus{}

	if utilfeature.DefaultFeatureGate.Enabled(features.PersistentVolumeLastPhaseTransitionTime) {
		pv.Status.Phase = api.VolumePending
		now := NowFunc()
		pv.Status.LastPhaseTransitionTime = &now
	}
}

func (persistentvolumeStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	persistentvolume := obj.(*api.PersistentVolume)
	opts := validation.ValidationOptionsForPersistentVolume(persistentvolume, nil)
	errorList := validation.ValidatePersistentVolume(persistentvolume, opts)
	return append(errorList, volumevalidation.ValidatePersistentVolume(persistentvolume)...)
}

// WarningsOnCreate returns warnings for the creation of the given object.
func (persistentvolumeStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return pvutil.GetWarningsForPersistentVolume(obj.(*api.PersistentVolume))
}

// Canonicalize normalizes the object after validation.
func (persistentvolumeStrategy) Canonicalize(obj runtime.Object) {
}

func (persistentvolumeStrategy) AllowCreateOnUpdate() bool {
	return false
}

// PrepareForUpdate sets the Status fields which is not allowed to be set by an end user updating a PV
func (persistentvolumeStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newPv := obj.(*api.PersistentVolume)
	oldPv := old.(*api.PersistentVolume)
	newPv.Status = oldPv.Status
}

func (persistentvolumeStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	newPv := obj.(*api.PersistentVolume)
	oldPv := old.(*api.PersistentVolume)
	opts := validation.ValidationOptionsForPersistentVolume(newPv, oldPv)
	errorList := validation.ValidatePersistentVolume(newPv, opts)
	errorList = append(errorList, volumevalidation.ValidatePersistentVolume(newPv)...)
	return append(errorList, validation.ValidatePersistentVolumeUpdate(newPv, oldPv, opts)...)
}

// WarningsOnUpdate returns warnings for the given update.
func (persistentvolumeStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return pvutil.GetWarningsForPersistentVolume(obj.(*api.PersistentVolume))
}

func (persistentvolumeStrategy) AllowUnconditionalUpdate() bool {
	return true
}

type persistentvolumeStatusStrategy struct {
	persistentvolumeStrategy
}

var StatusStrategy = persistentvolumeStatusStrategy{Strategy}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (persistentvolumeStatusStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"v1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("spec"),
		),
	}

	return fields
}

var NowFunc = metav1.Now

// PrepareForUpdate sets the Spec field which is not allowed to be changed when updating a PV's Status
func (persistentvolumeStatusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newPv := obj.(*api.PersistentVolume)
	oldPv := old.(*api.PersistentVolume)
	newPv.Spec = oldPv.Spec

	if utilfeature.DefaultFeatureGate.Enabled(features.PersistentVolumeLastPhaseTransitionTime) {
		switch {
		case oldPv.Status.Phase == newPv.Status.Phase && newPv.Status.LastPhaseTransitionTime == nil:
			// phase didn't change, preserve the existing transition time if set
			newPv.Status.LastPhaseTransitionTime = oldPv.Status.LastPhaseTransitionTime

		case oldPv.Status.Phase != newPv.Status.Phase && (newPv.Status.LastPhaseTransitionTime == nil || newPv.Status.LastPhaseTransitionTime.Equal(oldPv.Status.LastPhaseTransitionTime)):
			// phase changed and client didn't set or didn't change the transition time
			now := NowFunc()
			newPv.Status.LastPhaseTransitionTime = &now
		}
	}

	pvutil.DropDisabledStatusFields(&oldPv.Status, &newPv.Status)
}

func (persistentvolumeStatusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidatePersistentVolumeStatusUpdate(obj.(*api.PersistentVolume), old.(*api.PersistentVolume))
}

// WarningsOnUpdate returns warnings for the given update.
func (persistentvolumeStatusStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}
