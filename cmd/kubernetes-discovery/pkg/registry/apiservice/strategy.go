/*
Copyright 2016 The Kubernetes Authors.

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

package apiservice

import (
	"fmt"

	kapi "k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
	"k8s.io/kubernetes/pkg/util/validation/field"

	"k8s.io/kubernetes/cmd/kubernetes-discovery/pkg/apis/apiregistration"
	"k8s.io/kubernetes/cmd/kubernetes-discovery/pkg/apis/apiregistration/validation"
)

type apiServerStrategy struct {
	runtime.ObjectTyper
	kapi.NameGenerator
}

var strategy = apiServerStrategy{kapi.Scheme, kapi.SimpleNameGenerator}

func (apiServerStrategy) NamespaceScoped() bool {
	return false
}

func (apiServerStrategy) PrepareForCreate(ctx kapi.Context, obj runtime.Object) {
	_ = obj.(*apiregistration.APIService)
}

func (apiServerStrategy) PrepareForUpdate(ctx kapi.Context, obj, old runtime.Object) {
	newAPIService := obj.(*apiregistration.APIService)
	oldAPIService := old.(*apiregistration.APIService)
	newAPIService.Status = oldAPIService.Status
}

func (apiServerStrategy) Validate(ctx kapi.Context, obj runtime.Object) field.ErrorList {
	return validation.ValidateAPIService(obj.(*apiregistration.APIService))
}

func (apiServerStrategy) AllowCreateOnUpdate() bool {
	return false
}

func (apiServerStrategy) AllowUnconditionalUpdate() bool {
	return false
}

func (apiServerStrategy) Canonicalize(obj runtime.Object) {
}

func (apiServerStrategy) ValidateUpdate(ctx kapi.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateAPIServiceUpdate(obj.(*apiregistration.APIService), old.(*apiregistration.APIService))
}

// MatchAPIService is the filter used by the generic etcd backend to watch events
// from etcd to clients of the apiserver only interested in specific labels/fields.
func MatchAPIService(label labels.Selector, field fields.Selector) storage.SelectionPredicate {
	return storage.SelectionPredicate{
		Label: label,
		Field: field,
		GetAttrs: func(obj runtime.Object) (labels.Set, fields.Set, error) {
			apiserver, ok := obj.(*apiregistration.APIService)
			if !ok {
				return nil, nil, fmt.Errorf("given object is not a APIService.")
			}
			return labels.Set(apiserver.ObjectMeta.Labels), APIServiceToSelectableFields(apiserver), nil
		},
	}
}

// APIServiceToSelectableFields returns a field set that represents the object.
func APIServiceToSelectableFields(obj *apiregistration.APIService) fields.Set {
	return generic.ObjectMetaFieldsSet(&obj.ObjectMeta, true)
}
