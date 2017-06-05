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

package podpreset

import (
	"fmt"

	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	apistorage "k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/settings"
	"k8s.io/kubernetes/pkg/apis/settings/validation"
)

// podPresetStrategy implements verification logic for Pod Injection Policies.
type podPresetStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating Pod Injection Policy objects.
var Strategy = podPresetStrategy{api.Scheme, names.SimpleNameGenerator}

// NamespaceScoped returns true because all Pod Injection Policies need to be within a namespace.
func (podPresetStrategy) NamespaceScoped() bool {
	return true
}

// PrepareForCreate clears the status of a Pod Injection Policy before creation.
func (podPresetStrategy) PrepareForCreate(ctx genericapirequest.Context, obj runtime.Object) {
	pip := obj.(*settings.PodPreset)
	pip.Generation = 1
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (podPresetStrategy) PrepareForUpdate(ctx genericapirequest.Context, obj, old runtime.Object) {
	newPodPreset := obj.(*settings.PodPreset)
	oldPodPreset := old.(*settings.PodPreset)

	// Update is not allowed
	newPodPreset.Spec = oldPodPreset.Spec
}

// Validate validates a new PodPreset.
func (podPresetStrategy) Validate(ctx genericapirequest.Context, obj runtime.Object) field.ErrorList {
	pip := obj.(*settings.PodPreset)
	return validation.ValidatePodPreset(pip)
}

// Canonicalize normalizes the object after validation.
func (podPresetStrategy) Canonicalize(obj runtime.Object) {}

// AllowCreateOnUpdate is false for PodPreset; this means POST is needed to create one.
func (podPresetStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (podPresetStrategy) ValidateUpdate(ctx genericapirequest.Context, obj, old runtime.Object) field.ErrorList {
	validationErrorList := validation.ValidatePodPreset(obj.(*settings.PodPreset))
	updateErrorList := validation.ValidatePodPresetUpdate(obj.(*settings.PodPreset), old.(*settings.PodPreset))
	return append(validationErrorList, updateErrorList...)
}

// AllowUnconditionalUpdate is the default update policy for Pod Injection Policy objects.
func (podPresetStrategy) AllowUnconditionalUpdate() bool {
	return true
}

// SelectableFields returns a field set that represents the object.
func SelectableFields(pip *settings.PodPreset) fields.Set {
	return generic.ObjectMetaFieldsSet(&pip.ObjectMeta, true)
}

// GetAttrs returns labels and fields of a given object for filtering purposes.
func GetAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	pip, ok := obj.(*settings.PodPreset)
	if !ok {
		return nil, nil, fmt.Errorf("given object is not a PodPreset.")
	}
	return labels.Set(pip.ObjectMeta.Labels), SelectableFields(pip), nil
}

// Matcher is the filter used by the generic etcd backend to watch events
// from etcd to clients of the apiserver only interested in specific labels/fields.
func Matcher(label labels.Selector, field fields.Selector) apistorage.SelectionPredicate {
	return apistorage.SelectionPredicate{
		Label:    label,
		Field:    field,
		GetAttrs: GetAttrs,
	}
}
