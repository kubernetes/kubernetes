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

package flunder

import (
	"fmt"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/names"

	"k8s.io/sample-apiserver/pkg/apis/wardle"
)

func NewStrategy(typer runtime.ObjectTyper) flunderStrategy {
	return flunderStrategy{typer, names.SimpleNameGenerator}
}

func GetAttrs(obj runtime.Object) (labels.Set, fields.Set, bool, error) {
	apiserver, ok := obj.(*wardle.Flunder)
	if !ok {
		return nil, nil, false, fmt.Errorf("given object is not a Flunder.")
	}
	return labels.Set(apiserver.ObjectMeta.Labels), FlunderToSelectableFields(apiserver), apiserver.Initializers != nil, nil
}

// MatchFlunder is the filter used by the generic etcd backend to watch events
// from etcd to clients of the apiserver only interested in specific labels/fields.
func MatchFlunder(label labels.Selector, field fields.Selector) storage.SelectionPredicate {
	return storage.SelectionPredicate{
		Label:    label,
		Field:    field,
		GetAttrs: GetAttrs,
	}
}

// FlunderToSelectableFields returns a field set that represents the object.
func FlunderToSelectableFields(obj *wardle.Flunder) fields.Set {
	return generic.ObjectMetaFieldsSet(&obj.ObjectMeta, true)
}

type flunderStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// NamespaceScoped returns true because all Flunders need to be within a namespace.
func (flunderStrategy) NamespaceScoped() bool {
	return true
}

// PrepareForCreate clears the status of a Flunder before creation.
func (flunderStrategy) PrepareForCreate(ctx genericapirequest.Context, obj runtime.Object) {
	flunder := obj.(*wardle.Flunder)

	// create cannot set status
	flunder.Status = wardle.FlunderStatus{}
	flunder.Generation = 1
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (flunderStrategy) PrepareForUpdate(ctx genericapirequest.Context, obj, old runtime.Object) {
	newFlunder := obj.(*wardle.Flunder)
	oldFlunder := old.(*wardle.Flunder)

	// Update is not allowed to set status
	newFlunder.Status = oldFlunder.Status

	// Any changes to the spec increment the generation number, any changes to the
	// status should reflect the generation number of the corresponding object.
	// See metav1.ObjectMeta description for more information on Generation.
	if !apiequality.Semantic.DeepEqual(oldFlunder.Spec, newFlunder.Spec) {
		newFlunder.Generation = oldFlunder.Generation + 1
	}
}

// Validate validates a new Flunder.
func (flunderStrategy) Validate(ctx genericapirequest.Context, obj runtime.Object) field.ErrorList {
	return field.ErrorList{}
}

// AllowCreateOnUpdate is false for Flunder; this means POST is needed to create one.
func (flunderStrategy) AllowCreateOnUpdate() bool {
	return false
}

// AllowUnconditionalUpdate is the default update policy for Flunder objects.
func (flunderStrategy) AllowUnconditionalUpdate() bool {
	return false
}

// Canonicalize normalizes the object after validation.
func (flunderStrategy) Canonicalize(obj runtime.Object) {
}

// ValidateUpdate is the default update validation for an end user.
func (flunderStrategy) ValidateUpdate(ctx genericapirequest.Context, obj, old runtime.Object) field.ErrorList {
	return field.ErrorList{}
}

type flunderStatusStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

func NewFlunderStatusStrategy(typer runtime.ObjectTyper) flunderStatusStrategy {
	return flunderStatusStrategy{typer, names.SimpleNameGenerator}
}

func (flunderStatusStrategy) NamespaceScoped() bool {
	return false
}

func (flunderStatusStrategy) AllowCreateOnUpdate() bool {
	return false
}

func (flunderStatusStrategy) AllowUnconditionalUpdate() bool {
	return false
}

func (flunderStatusStrategy) Canonicalize(obj runtime.Object) {
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update of status.
func (flunderStatusStrategy) PrepareForUpdate(ctx genericapirequest.Context, obj, old runtime.Object) {
	newObj := obj.(*wardle.Flunder)
	oldObj := old.(*wardle.Flunder)
	newObj.Spec = oldObj.Spec

	// Status updates are for only for updating status, not objectmeta.
	newObj.Labels = oldObj.Labels
	newObj.Annotations = oldObj.Annotations
	newObj.OwnerReferences = oldObj.OwnerReferences
	newObj.Generation = oldObj.Generation
	newObj.SelfLink = oldObj.SelfLink
}

// ValidateUpdate is the default update validation for an end user updating status.
func (flunderStatusStrategy) ValidateUpdate(ctx genericapirequest.Context, obj, old runtime.Object) field.ErrorList {
	return field.ErrorList{}
}
