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

package component

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"
)

// Registry is a Storage-like interface with a Component-typed API.
type Registry interface {
	ListComponents(api.Context, labels.Selector, fields.Selector) (*api.ComponentList, error)
	GetComponent(ctx api.Context, name string) (*api.Component, error)
	CreateComponent(api.Context, *api.Component) (*api.Component, error)
	UpdateComponent(api.Context, *api.Component) (*api.Component, error)
	DeleteComponent(ctx api.Context, name string, options *api.DeleteOptions) error
	WatchComponents(ctx api.Context, lSelector labels.Selector, fSelector fields.Selector, resourceVersion string) (watch.Interface, error)
}

type StandardStorageRegistry interface {
	rest.StandardStorage
	Registry
}

// storage adds strongly typed storage methods to a component StandardStorage
type registry struct {
	rest.StandardStorage
}

// NewRegistry returns a new component Registry backed by the provided Storage. Any mismatched types will panic.
func NewRegistry(s rest.StandardStorage) StandardStorageRegistry {
	return &registry{s}
}

func (r *registry) ListComponents(ctx api.Context, labels labels.Selector, fields fields.Selector) (*api.ComponentList, error) {
	oList, err := r.List(ctx, labels, fields)
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve component list: %v", err)
	}
	return oList.(*api.ComponentList), nil
}

func (r *registry) GetComponent(ctx api.Context, name string) (*api.Component, error) {
	obj, err := r.Get(ctx, name)
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve component by name (%s): %v", name, err)
	}
	return obj.(*api.Component), nil
}

func (r *registry) CreateComponent(ctx api.Context, component *api.Component) (*api.Component, error) {
	obj, err := r.Create(ctx, component)
	if err != nil {
		return nil, fmt.Errorf("failed to create component: %v", err)
	}
	return obj.(*api.Component), nil
}

func (r *registry) UpdateComponent(ctx api.Context, component *api.Component) (*api.Component, error) {
	obj, _, err := r.Update(ctx, component)
	if err != nil {
		return nil, fmt.Errorf("failed to update component: %v", err)
	}
	return obj.(*api.Component), nil
}

func (r *registry) DeleteComponent(ctx api.Context, name string, options *api.DeleteOptions) error {
	_, err := r.Delete(ctx, name, options)
	if err != nil {
		return fmt.Errorf("failed to delete component: %v", err)
	}
	return nil
}

func (r *registry) WatchComponents(ctx api.Context, label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	iface, err := r.Watch(ctx, label, field, resourceVersion)
	if err != nil {
		return nil, fmt.Errorf("failed to watch component: %v", err)
	}
	return iface, nil
}
