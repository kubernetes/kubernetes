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

package service

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/watch"
)

// Registry is an interface for things that know how to store services.
type Registry interface {
	ListServices(ctx api.Context, options *api.ListOptions) (*api.ServiceList, error)
	CreateService(ctx api.Context, svc *api.Service) (*api.Service, error)
	GetService(ctx api.Context, name string) (*api.Service, error)
	DeleteService(ctx api.Context, name string) error
	UpdateService(ctx api.Context, svc *api.Service) (*api.Service, error)
	WatchServices(ctx api.Context, options *api.ListOptions) (watch.Interface, error)
	ExportService(ctx api.Context, name string, options unversioned.ExportOptions) (*api.Service, error)
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

func (s *storage) ListServices(ctx api.Context, options *api.ListOptions) (*api.ServiceList, error) {
	obj, err := s.List(ctx, options)
	if err != nil {
		return nil, err
	}
	return obj.(*api.ServiceList), nil
}

func (s *storage) CreateService(ctx api.Context, svc *api.Service) (*api.Service, error) {
	obj, err := s.Create(ctx, svc)
	if err != nil {
		return nil, err
	}
	return obj.(*api.Service), nil
}

func (s *storage) GetService(ctx api.Context, name string) (*api.Service, error) {
	obj, err := s.Get(ctx, name)
	if err != nil {
		return nil, err
	}
	return obj.(*api.Service), nil
}

func (s *storage) DeleteService(ctx api.Context, name string) error {
	_, err := s.Delete(ctx, name, nil)
	return err
}

func (s *storage) UpdateService(ctx api.Context, svc *api.Service) (*api.Service, error) {
	obj, _, err := s.Update(ctx, svc.Name, rest.DefaultUpdatedObjectInfo(svc, api.Scheme))
	if err != nil {
		return nil, err
	}
	return obj.(*api.Service), nil
}

func (s *storage) WatchServices(ctx api.Context, options *api.ListOptions) (watch.Interface, error) {
	return s.Watch(ctx, options)
}

// If StandardStorage implements rest.Exporter, returns exported service.
// Otherwise export is not supported.
func (s *storage) ExportService(ctx api.Context, name string, options unversioned.ExportOptions) (*api.Service, error) {
	exporter, isExporter := s.StandardStorage.(rest.Exporter)
	if !isExporter {
		return nil, fmt.Errorf("export is not supported")
	}
	obj, err := exporter.Export(ctx, name, options)
	if err != nil {
		return nil, err
	}
	return obj.(*api.Service), nil
}
