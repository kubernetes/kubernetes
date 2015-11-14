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

package service

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

// svcStrategy implements behavior for Services
type svcStrategy struct {
	runtime.ObjectTyper
	api.NameGenerator
}

// Services is the default logic that applies when creating and updating Service
// objects.
var Strategy = svcStrategy{api.Scheme, api.SimpleNameGenerator}

// NamespaceScoped is true for services.
func (svcStrategy) NamespaceScoped() bool {
	return true
}

// PrepareForCreate clears fields that are not allowed to be set by end users on creation.
func (svcStrategy) PrepareForCreate(obj runtime.Object) {
	service := obj.(*api.Service)
	service.Status = api.ServiceStatus{}
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (svcStrategy) PrepareForUpdate(obj, old runtime.Object) {
	// TODO: once service has a status sub-resource we can enable this.
	//newService := obj.(*api.Service)
	//oldService := old.(*api.Service)
	//newService.Status = oldService.Status
}

// Validate validates a new service.
func (svcStrategy) Validate(ctx api.Context, obj runtime.Object) fielderrors.ValidationErrorList {
	service := obj.(*api.Service)
	return validation.ValidateService(service)
}

// Canonicalize normalizes the object after validation.
func (svcStrategy) Canonicalize(obj runtime.Object) {
}

func (svcStrategy) AllowCreateOnUpdate() bool {
	return true
}

func (svcStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) fielderrors.ValidationErrorList {
	return validation.ValidateServiceUpdate(obj.(*api.Service), old.(*api.Service))
}

func (svcStrategy) AllowUnconditionalUpdate() bool {
	return true
}

func MatchServices(label labels.Selector, field fields.Selector) generic.Matcher {
	return &generic.SelectionPredicate{
		Label: label,
		Field: field,
		GetAttrs: func(obj runtime.Object) (labels.Set, fields.Set, error) {
			service, ok := obj.(*api.Service)
			if !ok {
				return nil, nil, fmt.Errorf("Given object is not a service")
			}
			return labels.Set(service.ObjectMeta.Labels), ServiceToSelectableFields(service), nil
		},
	}
}

func ServiceToSelectableFields(service *api.Service) fields.Set {
	return generic.ObjectMetaFieldsSet(service.ObjectMeta, true)
}
