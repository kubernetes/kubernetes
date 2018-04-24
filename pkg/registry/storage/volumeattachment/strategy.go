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

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/storage"
	"k8s.io/kubernetes/pkg/apis/storage/validation"
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

// ResetBeforeCreate clears the Status field which is not allowed to be set by end users on creation.
func (volumeAttachmentStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
}

func (volumeAttachmentStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	volumeAttachment := obj.(*storage.VolumeAttachment)
	return validation.ValidateVolumeAttachment(volumeAttachment)
}

// Canonicalize normalizes the object after validation.
func (volumeAttachmentStrategy) Canonicalize(obj runtime.Object) {
}

func (volumeAttachmentStrategy) AllowCreateOnUpdate() bool {
	return false
}

// PrepareForUpdate sets the Status fields which is not allowed to be set by an end user updating a PV
func (volumeAttachmentStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
}

func (volumeAttachmentStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	newVolumeAttachmentObj := obj.(*storage.VolumeAttachment)
	oldVolumeAttachmentObj := old.(*storage.VolumeAttachment)
	errorList := validation.ValidateVolumeAttachment(newVolumeAttachmentObj)
	return append(errorList, validation.ValidateVolumeAttachmentUpdate(newVolumeAttachmentObj, oldVolumeAttachmentObj)...)
}

func (volumeAttachmentStrategy) AllowUnconditionalUpdate() bool {
	return false
}
