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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"
)

type storage struct {
	registry Registry
}

// NewStorage returns a new StandardStorage backed by the provided component Registry
func NewStorage(registry Registry) rest.StandardStorage {
	return &storage{
		registry: registry,
	}
}

// New returns an empty Component.
// Satisfies the rest.Storage interface.
func (cs *storage) New() runtime.Object {
	return &api.Component{}
}

// NewList returns an empty ComponentList.
// Satisfies the rest.Lister interface.
func (cs *storage) NewList() runtime.Object {
	return &api.ComponentList{}
}

// List returns a list of component records that match to the label and field selectocs.
// Satisfies the rest.Lister interface.
func (cs *storage) List(ctx api.Context, labels labels.Selector, fields fields.Selector) (runtime.Object, error) {
	list, err := cs.registry.ListComponents(ctx, labels, fields)
	if err != nil {
		return nil, err
	}
	return list, err
}

// Get finds a component record by name and returns it.
// Satisfies the rest.Getter interface.
func (cs *storage) Get(ctx api.Context, name string) (runtime.Object, error) {
	component, err := cs.registry.GetComponent(ctx, name)
	if err != nil {
		return nil, err
	}
	return component, err
}

// Create initializes a component record, given a component object with type and url.
// A new unique name is generated and included in the returned component object.
// Satisfies the rest.Creater interface.
func (cs *storage) Create(ctx api.Context, obj runtime.Object) (runtime.Object, error) {
	component := obj.(*api.Component)

	if err := rest.BeforeCreate(CreateStrategy, ctx, obj); err != nil {
		return nil, err
	}

	out, err := cs.registry.CreateComponent(ctx, component)
	if err != nil {
		err = rest.CheckGeneratedNameError(CreateStrategy, err, component)
	}

	return out, err
}

// Update finds a component record by name and updates it.
// Satisfies the rest.Updater interface.
func (cs *storage) Update(ctx api.Context, obj runtime.Object) (runtime.Object, bool, error) {
	component := obj.(*api.Component)

	oldComponent, err := cs.registry.GetComponent(ctx, component.Name)
	if err != nil {
		return nil, false, err
	}

	if errs := validation.ValidateComponentUpdate(oldComponent, component); len(errs) > 0 {
		return nil, false, errors.NewInvalid("component", component.Name, errs)
	}

	out, err := cs.registry.UpdateComponent(ctx, component)

	// update never creates a new obj
	return out, false, err
}

// Delete finds a component record and deletes it.
// Satisfies the rest.GracefulDeleter interface
func (cs *storage) Delete(ctx api.Context, name string, options *api.DeleteOptions) (runtime.Object, error) {
	_, err := cs.registry.GetComponent(ctx, name)
	if err != nil {
		return nil, err
	}

	err = cs.registry.DeleteComponent(ctx, name, options)
	if err != nil {
		return nil, err
	}

	return &api.Status{Status: api.StatusSuccess}, nil
}

// Watch finds a component record and watches it for changes.
// Satisfies the rest.Watcher interface
func (cs *storage) Watch(ctx api.Context, label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return cs.registry.WatchComponents(ctx, label, field, resourceVersion)
}
