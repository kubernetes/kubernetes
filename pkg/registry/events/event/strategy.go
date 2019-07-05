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

package event

import (
	"context"
	"fmt"

	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/registry/generic"
	apistorage "k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/validation"
)

// eventStrategy implements verification logic for Pod Presets.
type eventStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating Pod Preset objects.
var Strategy = eventStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

// NamespaceScoped returns true because all Events need to be within a namespace.
func (eventStrategy) NamespaceScoped() bool {
	return true
}

// PrepareForCreate clears the status of a Pod Preset before creation.
func (eventStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (eventStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
}

// Validate validates a new Event.
func (eventStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	event := obj.(*api.Event)
	return validation.ValidateEvent(event)
}

// Canonicalize normalizes the object after validation.
// AllowCreateOnUpdate is false for Event; this means POST is needed to create one.
func (eventStrategy) AllowCreateOnUpdate() bool {
	return false
}

func (eventStrategy) Canonicalize(obj runtime.Object) {}

// ValidateUpdate is the default update validation for an end user.
func (eventStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	event := obj.(*api.Event)
	return validation.ValidateEvent(event)
}

// AllowUnconditionalUpdate is the default update policy for Event objects.
func (eventStrategy) AllowUnconditionalUpdate() bool {
	return true
}

// SelectableFields returns a field set that represents the object.
func SelectableFields(pip *api.Event) fields.Set {
	return generic.ObjectMetaFieldsSet(&pip.ObjectMeta, true)
}

// GetAttrs returns labels and fields of a given object for filtering purposes.
func GetAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	pip, ok := obj.(*api.Event)
	if !ok {
		return nil, nil, fmt.Errorf("given object is not a Event")
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
