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

package endpoint

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"
)

// Registry is an interface for things that know how to store endpoints.
type Registry interface {
	ListEndpoints(ctx api.Context) (*api.EndpointsList, error)
	GetEndpoints(ctx api.Context, name string) (*api.Endpoints, error)
	WatchEndpoints(ctx api.Context, labels labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error)
	UpdateEndpoints(ctx api.Context, e *api.Endpoints) error
	DeleteEndpoints(ctx api.Context, name string) error
}

// storage puts strong typing around storage calls
type storage struct {
	rest.StandardStorage
}

// NewRegistry returns a new Registry interface for the given Storage. Any mismatched
// types will panic.
func NewRegistry(s rest.StandardStorage) Registry {
	return &storage{s}
}

func (s *storage) ListEndpoints(ctx api.Context) (*api.EndpointsList, error) {
	obj, err := s.List(ctx, labels.Everything(), fields.Everything())
	if err != nil {
		return nil, err
	}
	return obj.(*api.EndpointsList), nil
}

func (s *storage) WatchEndpoints(ctx api.Context, label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return s.Watch(ctx, label, field, resourceVersion)
}

func (s *storage) GetEndpoints(ctx api.Context, name string) (*api.Endpoints, error) {
	obj, err := s.Get(ctx, name)
	if err != nil {
		return nil, err
	}
	return obj.(*api.Endpoints), nil
}

func (s *storage) UpdateEndpoints(ctx api.Context, endpoints *api.Endpoints) error {
	_, _, err := s.Update(ctx, endpoints)
	return err
}

func (s *storage) DeleteEndpoints(ctx api.Context, name string) error {
	_, err := s.Delete(ctx, name, nil)
	return err
}
