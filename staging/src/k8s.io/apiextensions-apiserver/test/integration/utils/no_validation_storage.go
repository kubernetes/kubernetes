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

package utils

import (
	"context"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/registry/rest"
)

type noValidationCreateStrategy struct {
	rest.RESTCreateStrategy
}

type noValidationUpdateStrategy struct {
	rest.RESTUpdateStrategy
}

// NewStoreWrapperWithoutValidation Creates a wrapper store with validation disabled on both create and update/
// This store is useful for downgrade tests as the objects are not being validated on GET path and this will give
// a chance to tests to store invalid objects and test the handlers/controllers behaviour on that.
func NewStoreWrapperWithoutValidation(store *registry.Store) *registry.Store {
	ret := *store
	ret.CreateStrategy = &noValidationCreateStrategy{
		RESTCreateStrategy: ret.CreateStrategy,
	}
	ret.UpdateStrategy = &noValidationUpdateStrategy{
		RESTUpdateStrategy: ret.UpdateStrategy,
	}
	return &ret
}

func (s *noValidationCreateStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	return field.ErrorList{}
}

func (s *noValidationUpdateStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return field.ErrorList{}
}
