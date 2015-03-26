/*
Copyright 2014 Google Inc. All rights reserved.

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
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/fielderrors"
)

// persistentvolumeStrategy implements behavior for PersistentVolume objects
type persistentvolumeStrategy struct {
	runtime.ObjectTyper
	api.NameGenerator
}

// Strategy is the default logic that applies when creating and updating PersistentVolume
// objects via the REST API.
var Strategy = persistentvolumeStrategy{api.Scheme, api.SimpleNameGenerator}

// NamespaceScoped is false for persistentvolumes.
func (persistentvolumeStrategy) NamespaceScoped() bool {
	return false
}

// ResetBeforeCreate clears fields that are not allowed to be set by end users on creation.
func (persistentvolumeStrategy) PrepareForCreate(obj runtime.Object) {
	pv := obj.(*api.PersistentVolume)
	pv.Status = api.PersistentVolumeStatus{}
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (persistentvolumeStrategy) PrepareForUpdate(obj, old runtime.Object) {
	newPv := obj.(*api.PersistentVolume)
	oldPv := obj.(*api.PersistentVolume)
	newPv.Status = oldPv.Status
}

// Validate validates a new persistentvolume.
func (persistentvolumeStrategy) Validate(obj runtime.Object) fielderrors.ValidationErrorList {
	persistentvolume := obj.(*api.PersistentVolume)
	return validation.ValidatePersistentVolume(persistentvolume)
}

// AllowCreateOnUpdate is false for persistentvolumes.
func (persistentvolumeStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (persistentvolumeStrategy) ValidateUpdate(obj, old runtime.Object) fielderrors.ValidationErrorList {
	return validation.ValidatePersistentVolumeUpdate(obj.(*api.PersistentVolume), old.(*api.PersistentVolume))
}

type persistentvolumeStatusStrategy struct {
	persistentvolumeStrategy
}

var StatusStrategy = persistentvolumeStatusStrategy{Strategy}

func (persistentvolumeStatusStrategy) PrepareForUpdate(obj, old runtime.Object) {
	newPv := obj.(*api.PersistentVolume)
	oldPv := obj.(*api.PersistentVolume)
	newPv.Spec = oldPv.Spec
}

func (persistentvolumeStatusStrategy) ValidateUpdate(obj, old runtime.Object) fielderrors.ValidationErrorList {
	return validation.ValidatePersistentVolumeStatusUpdate(obj.(*api.PersistentVolume), old.(*api.PersistentVolume))
}

// MatchPersistentVolume returns a generic matcher for a given label and field selector.
func MatchPersistentVolumes(label labels.Selector, field fields.Selector) generic.Matcher {
	return generic.MatcherFunc(func(obj runtime.Object) (bool, error) {
		persistentvolumeObj, ok := obj.(*api.PersistentVolume)
		if !ok {
			return false, fmt.Errorf("not a persistentvolume")
		}
		fields := PersistentVolumeToSelectableFields(persistentvolumeObj)
		return label.Matches(labels.Set(persistentvolumeObj.Labels)) && field.Matches(fields), nil
	})
}

// PersistentVolumeToSelectableFields returns a label set that represents the object
// TODO: fields are not labels, and the validation rules for them do not apply.
func PersistentVolumeToSelectableFields(persistentvolume *api.PersistentVolume) labels.Set {
	return labels.Set{
		"name": persistentvolume.Name,
	}
}
