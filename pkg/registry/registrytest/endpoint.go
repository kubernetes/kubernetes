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

package registrytest

import (
	"context"
	"fmt"
	"sync"

	"k8s.io/apimachinery/pkg/api/errors"
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/registry/rest"
	api "k8s.io/kubernetes/pkg/apis/core"
)

// Registry is an interface for things that know how to store endpoints.
type EndpointRegistry struct {
	Endpoints *api.EndpointsList
	Updates   []api.Endpoints
	Err       error

	lock sync.Mutex
}

func (e *EndpointRegistry) List(ctx context.Context, options *metainternalversion.ListOptions) (runtime.Object, error) {
	// TODO: support namespaces in this mock
	e.lock.Lock()
	defer e.lock.Unlock()

	return e.Endpoints, e.Err
}

func (e *EndpointRegistry) New() runtime.Object {
	return &api.Endpoints{}
}
func (e *EndpointRegistry) NewList() runtime.Object {
	return &api.EndpointsList{}
}

func (e *EndpointRegistry) Get(ctx context.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	// TODO: support namespaces in this mock
	e.lock.Lock()
	defer e.lock.Unlock()
	if e.Err != nil {
		return nil, e.Err
	}
	if e.Endpoints != nil {
		for _, endpoint := range e.Endpoints.Items {
			if endpoint.Name == name {
				return &endpoint, nil
			}
		}
	}
	return nil, errors.NewNotFound(api.Resource("endpoints"), name)
}

func (e *EndpointRegistry) Watch(ctx context.Context, options *metainternalversion.ListOptions) (watch.Interface, error) {
	return nil, fmt.Errorf("unimplemented!")
}

func (e *EndpointRegistry) Create(ctx context.Context, endpoints runtime.Object, createValidation rest.ValidateObjectFunc, options *metav1.CreateOptions) (runtime.Object, error) {
	return nil, fmt.Errorf("unimplemented!")
}

func (e *EndpointRegistry) Update(ctx context.Context, name string, objInfo rest.UpdatedObjectInfo, createValidation rest.ValidateObjectFunc, updateValidation rest.ValidateObjectUpdateFunc, forceAllowCreateOnUpdate bool, options *metav1.UpdateOptions) (runtime.Object, bool, error) {
	obj, err := objInfo.UpdatedObject(ctx, nil)
	if err != nil {
		return nil, false, err
	}
	endpoints := obj.(*api.Endpoints)
	// TODO: support namespaces in this mock
	e.lock.Lock()
	defer e.lock.Unlock()

	e.Updates = append(e.Updates, *endpoints)

	if e.Err != nil {
		return nil, false, e.Err
	}
	if e.Endpoints == nil {
		e.Endpoints = &api.EndpointsList{
			Items: []api.Endpoints{
				*endpoints,
			},
		}
		return endpoints, false, nil
	}
	for ix := range e.Endpoints.Items {
		if e.Endpoints.Items[ix].Name == endpoints.Name {
			e.Endpoints.Items[ix] = *endpoints
		}
	}
	e.Endpoints.Items = append(e.Endpoints.Items, *endpoints)
	return endpoints, false, nil
}

func (e *EndpointRegistry) Delete(ctx context.Context, name string, deleteValidation rest.ValidateObjectFunc, options *metav1.DeleteOptions) (runtime.Object, bool, error) {
	// TODO: support namespaces in this mock
	e.lock.Lock()
	defer e.lock.Unlock()
	if e.Err != nil {
		return nil, false, e.Err
	}
	if e.Endpoints != nil {
		var newList []api.Endpoints
		for _, endpoint := range e.Endpoints.Items {
			if endpoint.Name != name {
				newList = append(newList, endpoint)
			}
		}
		e.Endpoints.Items = newList
	}
	return nil, true, nil
}

func (e *EndpointRegistry) DeleteCollection(ctx context.Context, _ rest.ValidateObjectFunc, _ *metav1.DeleteOptions, _ *metainternalversion.ListOptions) (runtime.Object, error) {
	return nil, fmt.Errorf("unimplemented!")
}
