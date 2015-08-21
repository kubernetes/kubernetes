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

package controller

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"
)

// Registry is an interface for things that know how to store ReplicationControllers.
type Registry interface {
	ListControllers(ctx api.Context, label labels.Selector, field fields.Selector) (*api.ReplicationControllerList, error)
	WatchControllers(ctx api.Context, label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error)
	GetController(ctx api.Context, controllerID string) (*api.ReplicationController, error)
	CreateController(ctx api.Context, controller *api.ReplicationController) (*api.ReplicationController, error)
	UpdateController(ctx api.Context, controller *api.ReplicationController) (*api.ReplicationController, error)
	DeleteController(ctx api.Context, controllerID string) error
}

// registry puts strong typing around storage calls
type registry struct {
	rest.StandardStorage
}

// NewRegistry returns a new Registry interface for the given Storage. Any mismatched
// types will panic.
func NewRegistry(s rest.StandardStorage) Registry {
	return &registry{s}
}

// List obtains a list of ReplicationControllers that match selector.
func (s *registry) ListControllers(ctx api.Context, label labels.Selector, field fields.Selector) (*api.ReplicationControllerList, error) {
	if !field.Empty() {
		return nil, fmt.Errorf("field selector not supported yet")
	}
	obj, err := s.List(ctx, label, field)
	if err != nil {
		return nil, err
	}
	return obj.(*api.ReplicationControllerList), err
}

func (s *registry) WatchControllers(ctx api.Context, label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return s.Watch(ctx, label, field, resourceVersion)
}

func (s *registry) GetController(ctx api.Context, controllerID string) (*api.ReplicationController, error) {
	obj, err := s.Get(ctx, controllerID)
	if err != nil {
		return nil, err
	}
	return obj.(*api.ReplicationController), nil
}

func (s *registry) CreateController(ctx api.Context, controller *api.ReplicationController) (*api.ReplicationController, error) {
	obj, err := s.Create(ctx, controller)
	if err != nil {
		return nil, err
	}
	return obj.(*api.ReplicationController), nil
}

func (s *registry) UpdateController(ctx api.Context, controller *api.ReplicationController) (*api.ReplicationController, error) {
	obj, _, err := s.Update(ctx, controller)
	if err != nil {
		return nil, err
	}
	return obj.(*api.ReplicationController), nil
}

func (s *registry) DeleteController(ctx api.Context, controllerID string) error {
	_, err := s.Delete(ctx, controllerID, nil)
	return err
}
