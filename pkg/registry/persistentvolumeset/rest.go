/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package persistentvolumeset

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/fielderrors"
)

// persistentVolumeSetStrategy implements behavior for PersistentVolumeSet objects
type persistentVolumeSetStrategy struct {
	runtime.ObjectTyper
	api.NameGenerator
}

// Strategy is the default logic that applies when creating and updating PersistentVolumeSet
// objects via the REST API.
var Strategy = persistentVolumeSetStrategy{api.Scheme, api.SimpleNameGenerator}

func (persistentVolumeSetStrategy) NamespaceScoped() bool {
	return false
}

// ResetBeforeCreate clears the Status field which is not allowed to be set by end users on creation.
func (persistentVolumeSetStrategy) PrepareForCreate(obj runtime.Object) {
	pv := obj.(*api.PersistentVolumeSet)
	pv.Status = api.PersistentVolumeSetStatus{}
}

func (persistentVolumeSetStrategy) Validate(ctx api.Context, obj runtime.Object) fielderrors.ValidationErrorList {
	persistentvolumeset := obj.(*api.PersistentVolumeSet)
	return validation.ValidatePersistentVolumeSet(persistentvolumeset)
}

func (persistentVolumeSetStrategy) AllowCreateOnUpdate() bool {
	return false
}

// PrepareForUpdate sets the Status fields which is not allowed to be set by an end user updating a PV
func (persistentVolumeSetStrategy) PrepareForUpdate(obj, old runtime.Object) {
	newPv := obj.(*api.PersistentVolumeSet)
	oldPv := obj.(*api.PersistentVolumeSet)
	newPv.Status = oldPv.Status
}

func (persistentVolumeSetStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) fielderrors.ValidationErrorList {
	errorList := validation.ValidatePersistentVolumeSet(obj.(*api.PersistentVolumeSet))
	return append(errorList, validation.ValidatePersistentVolumeSetUpdate(obj.(*api.PersistentVolumeSet), old.(*api.PersistentVolumeSet))...)
}

func (persistentVolumeSetStrategy) AllowUnconditionalUpdate() bool {
	return true
}

type persistentVolumeSetStatusStrategy struct {
	persistentVolumeSetStrategy
}

var StatusStrategy = persistentVolumeSetStatusStrategy{Strategy}

// PrepareForUpdate sets the Spec field which is not allowed to be changed when updating a PV's Status
func (persistentVolumeSetStatusStrategy) PrepareForUpdate(obj, old runtime.Object) {
	newPv := obj.(*api.PersistentVolumeSet)
	oldPv := obj.(*api.PersistentVolumeSet)
	newPv.Spec = oldPv.Spec
}

func (persistentVolumeSetStatusStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) fielderrors.ValidationErrorList {
	return validation.ValidatePersistentVolumeSetStatusUpdate(obj.(*api.PersistentVolumeSet), old.(*api.PersistentVolumeSet))
}

// MatchPersistentVolume returns a generic matcher for a given label and field selector.
func MatchPersistentVolumeSets(label labels.Selector, field fields.Selector) generic.Matcher {
	return generic.MatcherFunc(func(obj runtime.Object) (bool, error) {
		persistentvolumeObj, ok := obj.(*api.PersistentVolumeSet)
		if !ok {
			return false, fmt.Errorf("not a persistentvolumeset")
		}
		fields := PersistentVolumeSetToSelectableFields(persistentvolumeObj)
		return label.Matches(labels.Set(persistentvolumeObj.Labels)) && field.Matches(fields), nil
	})
}

// PersistentVolumeSetToSelectableFields returns a label set that represents the object
func PersistentVolumeSetToSelectableFields(persistentvolumeset *api.PersistentVolumeSet) labels.Set {
	return labels.Set{
		"metadata.name": persistentvolumeset.Name,
	}
}
