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

package rest

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/fielderrors"
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

// ResetBeforeCreate clears fields that are not allowed to be set by end users on creation.
func (svcStrategy) ResetBeforeCreate(obj runtime.Object) {
	service := obj.(*api.Service)
	service.Status = api.ServiceStatus{}
}

// Validate validates a new service.
func (svcStrategy) Validate(obj runtime.Object) fielderrors.ValidationErrorList {
	service := obj.(*api.Service)
	return validation.ValidateService(service)
}

func (svcStrategy) AllowCreateOnUpdate() bool {
	return true
}

func (svcStrategy) ValidateUpdate(obj, old runtime.Object) fielderrors.ValidationErrorList {
	return validation.ValidateServiceUpdate(old.(*api.Service), obj.(*api.Service))
}

// nodeStrategy implements behavior for nodes
// TODO: move to a node specific package.
type nodeStrategy struct {
	runtime.ObjectTyper
	api.NameGenerator
}

// Nodes is the default logic that applies when creating and updating Node
// objects.
var Nodes RESTCreateStrategy = nodeStrategy{api.Scheme, api.SimpleNameGenerator}

// NamespaceScoped is false for nodes.
func (nodeStrategy) NamespaceScoped() bool {
	return false
}

// ResetBeforeCreate clears fields that are not allowed to be set by end users on creation.
func (nodeStrategy) ResetBeforeCreate(obj runtime.Object) {
	_ = obj.(*api.Node)
	// Nodes allow *all* fields, including status, to be set.
}

// Validate validates a new node.
func (nodeStrategy) Validate(obj runtime.Object) fielderrors.ValidationErrorList {
	node := obj.(*api.Node)
	return validation.ValidateMinion(node)
}
