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
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/validation/field"
)

// persistentvolumeclaimStrategy implements behavior for PersistentVolumeClaim objects
type persistentvolumeclaimStrategy struct {
	runtime.ObjectTyper
	api.NameGenerator
}

// Strategy is the default logic that applies when creating and updating PersistentVolumeClaim
// objects via the REST API.
var Strategy = persistentvolumeclaimStrategy{api.Scheme, api.SimpleNameGenerator}

func (persistentvolumeclaimStrategy) NamespaceScoped() bool {
	return true
}

// PrepareForCreate clears the Status field which is not allowed to be set by end users on creation.
func (persistentvolumeclaimStrategy) PrepareForCreate(ctx api.Context, obj runtime.Object) {
	pv := obj.(*api.PersistentVolumeClaim)
	pv.Status = api.PersistentVolumeClaimStatus{}
}

func (persistentvolumeclaimStrategy) Validate(ctx api.Context, obj runtime.Object) field.ErrorList {
	pvc := obj.(*api.PersistentVolumeClaim)
	return validation.ValidatePersistentVolumeClaim(pvc)
}

// Canonicalize normalizes the object after validation.
func (persistentvolumeclaimStrategy) Canonicalize(obj runtime.Object) {
}

func (persistentvolumeclaimStrategy) AllowCreateOnUpdate() bool {
	return false
}

// PrepareForUpdate sets the Status field which is not allowed to be set by end users on update
func (persistentvolumeclaimStrategy) PrepareForUpdate(ctx api.Context, obj, old runtime.Object) {
	newPvc := obj.(*api.PersistentVolumeClaim)
	oldPvc := old.(*api.PersistentVolumeClaim)
	newPvc.Status = oldPvc.Status
}

func (persistentvolumeclaimStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) field.ErrorList {
	errorList := validation.ValidatePersistentVolumeClaim(obj.(*api.PersistentVolumeClaim))
	return append(errorList, validation.ValidatePersistentVolumeClaimUpdate(obj.(*api.PersistentVolumeClaim), old.(*api.PersistentVolumeClaim))...)
}

func (persistentvolumeclaimStrategy) AllowUnconditionalUpdate() bool {
	return true
}

type persistentvolumeclaimStatusStrategy struct {
	persistentvolumeclaimStrategy
}

var StatusStrategy = persistentvolumeclaimStatusStrategy{Strategy}

// PrepareForUpdate sets the Spec field which is not allowed to be changed when updating a PV's Status
func (persistentvolumeclaimStatusStrategy) PrepareForUpdate(ctx api.Context, obj, old runtime.Object) {
	newPv := obj.(*api.PersistentVolumeClaim)
	oldPv := old.(*api.PersistentVolumeClaim)
	newPv.Spec = oldPv.Spec
}

func (persistentvolumeclaimStatusStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) field.ErrorList {
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
func PersistentVolumeClaimToSelectableFields(persistentvolumeclaim *api.PersistentVolumeClaim) labels.Set {
	objectMetaFieldsSet := generic.ObjectMetaFieldsSet(persistentvolumeclaim.ObjectMeta, true)
	specificFieldsSet := fields.Set{
		// This is a bug, but we need to support it for backward compatibility.
		"name": persistentvolumeclaim.Name,
	}
	return labels.Set(generic.MergeFieldsSets(objectMetaFieldsSet, specificFieldsSet))
}
