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

package persistentvolumeclaim

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

// persistentvolumeclaimStrategy implements behavior for PersistentVolumeClaim objects
type persistentvolumeclaimStrategy struct {
	runtime.ObjectTyper
	api.NameGenerator
}

// Strategy is the default logic that applies when creating and updating PersistentVolumeClaim
// objects via the REST API.
var Strategy = persistentvolumeclaimStrategy{api.Scheme, api.SimpleNameGenerator}

// NamespaceScoped is true for persistentvolumeclaims.
func (persistentvolumeclaimStrategy) NamespaceScoped() bool {
	return true
}

// ResetBeforeCreate clears fields that are not allowed to be set by end users on creation.
func (persistentvolumeclaimStrategy) PrepareForCreate(obj runtime.Object) {
	pv := obj.(*api.PersistentVolumeClaim)
	pv.Status = api.PersistentVolumeClaimStatus{}
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (persistentvolumeclaimStrategy) PrepareForUpdate(obj, old runtime.Object) {
	newPvc := obj.(*api.PersistentVolumeClaim)
	oldPvc := obj.(*api.PersistentVolumeClaim)
	newPvc.Status = oldPvc.Status
}

// Validate validates a new persistentvolumeclaim.
func (persistentvolumeclaimStrategy) Validate(obj runtime.Object) fielderrors.ValidationErrorList {
	pvc := obj.(*api.PersistentVolumeClaim)
	return validation.ValidatePersistentVolumeClaim(pvc)
}

// AllowCreateOnUpdate is false for persistentvolumeclaims.
func (persistentvolumeclaimStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (persistentvolumeclaimStrategy) ValidateUpdate(obj, old runtime.Object) fielderrors.ValidationErrorList {
	return validation.ValidatePersistentVolumeClaimUpdate(obj.(*api.PersistentVolumeClaim), old.(*api.PersistentVolumeClaim))
}

type persistentvolumeclaimStatusStrategy struct {
	persistentvolumeclaimStrategy
}

var StatusStrategy = persistentvolumeclaimStatusStrategy{Strategy}

func (persistentvolumeclaimStatusStrategy) PrepareForUpdate(obj, old runtime.Object) {
	newPvc := obj.(*api.PersistentVolumeClaim)
	oldPvc := obj.(*api.PersistentVolumeClaim)
	newPvc.Spec = oldPvc.Spec
}

func (persistentvolumeclaimStatusStrategy) ValidateUpdate(obj, old runtime.Object) fielderrors.ValidationErrorList {
	return validation.ValidatePersistentVolumeClaimStatusUpdate(obj.(*api.PersistentVolumeClaim), old.(*api.PersistentVolumeClaim))
}

// MatchPersistentVolumeClaim returns a generic matcher for a given label and field selector.
func MatchPersistentVolumeClaim(label labels.Selector, field fields.Selector) generic.Matcher {
	return generic.MatcherFunc(func(obj runtime.Object) (bool, error) {
		persistentvolumeclaimObj, ok := obj.(*api.PersistentVolumeClaim)
		if !ok {
			return false, fmt.Errorf("not a persistentvolumeclaim")
		}
		fields := PersistentVolumeClaimToSelectableFields(persistentvolumeclaimObj)
		return label.Matches(labels.Set(persistentvolumeclaimObj.Labels)) && field.Matches(fields), nil
	})
}

// PersistentVolumeClaimToSelectableFields returns a label set that represents the object
// TODO: fields are not labels, and the validation rules for them do not apply.
func PersistentVolumeClaimToSelectableFields(persistentvolumeclaim *api.PersistentVolumeClaim) labels.Set {
	return labels.Set{
		"name": persistentvolumeclaim.Name,
	}
}
