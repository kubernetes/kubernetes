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

package rest

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/fielderrors"
)

// ObjectFunc is a function to act on a given object. An error may be returned
// if the hook cannot be completed. An ObjectFunc may transform the provided
// object.
type ObjectFunc func(obj runtime.Object) error

// AllFuncs returns an ObjectFunc that attempts to run all of the provided functions
// in order, returning early if there are any errors.
func AllFuncs(fns ...ObjectFunc) ObjectFunc {
	return func(obj runtime.Object) error {
		for _, fn := range fns {
			if fn == nil {
				continue
			}
			if err := fn(obj); err != nil {
				return err
			}
		}
		return nil
	}
}

// svcStrategy implements behavior for Services
// TODO: move to a service specific package.
type svcStrategy struct {
	runtime.ObjectTyper
	api.NameGenerator
}

// Services is the default logic that applies when creating and updating Service
// objects.
var Services = svcStrategy{api.Scheme, api.SimpleNameGenerator}

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

func (svcStrategy) AllowCreateOnUpdate() bool {
	return true
}

func (svcStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) fielderrors.ValidationErrorList {
	return validation.ValidateServiceUpdate(old.(*api.Service), obj.(*api.Service))
}

func (svcStrategy) AllowUnconditionalUpdate() bool {
	return true
}
