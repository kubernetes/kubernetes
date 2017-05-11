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

package wardle

import (
	"fmt"

	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/names"

	"k8s.io/apimachinery/pkg/apis/discovery"
)

type groupStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

func NewStrategy(typer runtime.ObjectTyper) groupStrategy {
	return groupStrategy{typer, names.SimpleNameGenerator}
}

func (groupStrategy) NamespaceScoped() bool {
	return false
}

func (groupStrategy) PrepareForCreate(ctx genericapirequest.Context, obj runtime.Object) {
}

func (groupStrategy) PrepareForUpdate(ctx genericapirequest.Context, obj, old runtime.Object) {
}

func (groupStrategy) Validate(ctx genericapirequest.Context, obj runtime.Object) field.ErrorList {
	return field.ErrorList{}
	// return validation.ValidateFlunder(obj.(*discovery.Group))
}

func (groupStrategy) AllowCreateOnUpdate() bool {
	return false
}

func (groupStrategy) AllowUnconditionalUpdate() bool {
	return false
}

func (groupStrategy) Canonicalize(obj runtime.Object) {
}

func (groupStrategy) ValidateUpdate(ctx genericapirequest.Context, obj, old runtime.Object) field.ErrorList {
	return field.ErrorList{}
	// return validation.ValidateGroupUpdate(obj.(*discovery.Group), old.(*discovery.Group))
}

func GetAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	apiserver, ok := obj.(*discovery.Group)
	if !ok {
		return nil, nil, fmt.Errorf("given object is not a Group.")
	}
	return labels.Set(apiserver.ObjectMeta.Labels), GroupToSelectableFields(apiserver), nil
}

// MatchGroup is the filter used by the generic etcd backend to watch events
// from etcd to clients of the apiserver only interested in specific labels/fields.
func MatchGroup(label labels.Selector, field fields.Selector) storage.SelectionPredicate {
	return storage.SelectionPredicate{
		Label:    label,
		Field:    field,
		GetAttrs: GetAttrs,
	}
}

// GroupToSelectableFields returns a field set that represents the object.
func GroupToSelectableFields(obj *discovery.Group) fields.Set {
	return generic.ObjectMetaFieldsSet(&obj.ObjectMeta, true)
}
