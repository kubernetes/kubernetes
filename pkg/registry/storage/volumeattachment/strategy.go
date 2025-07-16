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

package volumeattachment

import (
	"context"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/storage"
	"k8s.io/kubernetes/pkg/apis/storage/validation"
	"k8s.io/kubernetes/pkg/features"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
)

// volumeAttachmentStrategy implements behavior for VolumeAttachment objects
type volumeAttachmentStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating
// VolumeAttachment objects via the REST API.
var Strategy = volumeAttachmentStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

func (volumeAttachmentStrategy) NamespaceScoped() bool {
	return false
}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (volumeAttachmentStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"storage.k8s.io/v1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
	}

	return fields
}

// ResetBeforeCreate clears the Status field which is not allowed to be set by end users on creation.
func (volumeAttachmentStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	volumeAttachment := obj.(*storage.VolumeAttachment)
	volumeAttachment.Status = storage.VolumeAttachmentStatus{}
}

func (volumeAttachmentStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	volumeAttachment := obj.(*storage.VolumeAttachment)

	errs := validation.ValidateVolumeAttachment(volumeAttachment)

	// tighten up validation of newly created v1 attachments
	errs = append(errs, validation.ValidateVolumeAttachmentV1(volumeAttachment)...)
	return errs
}

// WarningsOnCreate returns warnings for the creation of the given object.
func (volumeAttachmentStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

// Canonicalize normalizes the object after validation.
func (volumeAttachmentStrategy) Canonicalize(obj runtime.Object) {
}

func (volumeAttachmentStrategy) AllowCreateOnUpdate() bool {
	return false
}

// PrepareForUpdate sets the Status fields which is not allowed to be set by an end user updating a VolumeAttachment
func (volumeAttachmentStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newVolumeAttachment := obj.(*storage.VolumeAttachment)
	oldVolumeAttachment := old.(*storage.VolumeAttachment)

	newVolumeAttachment.Status = oldVolumeAttachment.Status
	// No need to increment Generation because we don't allow updates to spec

}

func (volumeAttachmentStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	newVolumeAttachmentObj := obj.(*storage.VolumeAttachment)
	oldVolumeAttachmentObj := old.(*storage.VolumeAttachment)
	errorList := validation.ValidateVolumeAttachment(newVolumeAttachmentObj)
	return append(errorList, validation.ValidateVolumeAttachmentUpdate(newVolumeAttachmentObj, oldVolumeAttachmentObj)...)
}

// WarningsOnUpdate returns warnings for the given update.
func (volumeAttachmentStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func (volumeAttachmentStrategy) AllowUnconditionalUpdate() bool {
	return false
}

// volumeAttachmentStatusStrategy implements behavior for VolumeAttachmentStatus subresource
type volumeAttachmentStatusStrategy struct {
	volumeAttachmentStrategy
}

// StatusStrategy is the default logic that applies when creating and updating
// VolumeAttachmentStatus subresource via the REST API.
var StatusStrategy = volumeAttachmentStatusStrategy{Strategy}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (volumeAttachmentStatusStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"storage.k8s.io/v1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("metadata"),
			fieldpath.MakePathOrDie("spec"),
		),
	}

	return fields
}

// PrepareForUpdate sets the Status fields which is not allowed to be set by an end user updating a VolumeAttachment
func (volumeAttachmentStatusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newVolumeAttachment := obj.(*storage.VolumeAttachment)
	oldVolumeAttachment := old.(*storage.VolumeAttachment)

	newVolumeAttachment.Spec = oldVolumeAttachment.Spec
	metav1.ResetObjectMetaForStatus(&newVolumeAttachment.ObjectMeta, &oldVolumeAttachment.ObjectMeta)

	if !feature.DefaultFeatureGate.Enabled(features.MutableCSINodeAllocatableCount) {
		// Only clear ErrorCode field if it isn't set in the old object
		if newVolumeAttachment.Status.AttachError != nil {
			if oldVolumeAttachment.Status.AttachError == nil || oldVolumeAttachment.Status.AttachError.ErrorCode == nil {
				newVolumeAttachment.Status.AttachError.ErrorCode = nil
			}
		}
		if newVolumeAttachment.Status.DetachError != nil {
			if oldVolumeAttachment.Status.DetachError == nil || oldVolumeAttachment.Status.DetachError.ErrorCode == nil {
				newVolumeAttachment.Status.DetachError.ErrorCode = nil
			}
		}
	}
}
