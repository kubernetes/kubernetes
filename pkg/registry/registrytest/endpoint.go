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

func (e *EndpointRegistry) ListEndpoints(ctx context.Context, options *metainternalversion.ListOptions) (*api.EndpointsList, error) {
	// TODO: support namespaces in this mock
	e.lock.Lock()
	defer e.lock.Unlock()

	return e.Endpoints, e.Err
}

func (e *EndpointRegistry) GetEndpoints(ctx context.Context, name string, options *metav1.GetOptions) (*api.Endpoints, error) {
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

func (e *EndpointRegistry) WatchEndpoints(ctx context.Context, options *metainternalversion.ListOptions) (watch.Interface, error) {
	return nil, fmt.Errorf("unimplemented!")
}

func (e *EndpointRegistry) UpdateEndpoints(ctx context.Context, endpoints *api.Endpoints, createValidation rest.ValidateObjectFunc, updateValidation rest.ValidateObjectUpdateFunc) error {
	// TODO: support namespaces in this mock
	e.lock.Lock()
	defer e.lock.Unlock()

	e.Updates = append(e.Updates, *endpoints)

	if e.Err != nil {
		return e.Err
	}
	if e.Endpoints == nil {
		e.Endpoints = &api.EndpointsList{
			Items: []api.Endpoints{
				*endpoints,
			},
		}
		return nil
	}
	for ix := range e.Endpoints.Items {
		if e.Endpoints.Items[ix].Name == endpoints.Name {
			e.Endpoints.Items[ix] = *endpoints
		}
	}
	e.Endpoints.Items = append(e.Endpoints.Items, *endpoints)
	return nil
}

func (e *EndpointRegistry) DeleteEndpoints(ctx context.Context, name string) error {
	// TODO: support namespaces in this mock
	e.lock.Lock()
	defer e.lock.Unlock()
	if e.Err != nil {
		return e.Err
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
	return nil
}
