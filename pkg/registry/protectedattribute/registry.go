/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package protectedattribute

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/watch"
)

// Registry interface exposes operations on ProtectedAttribute's.
type Registry interface {
	ListProtectedAttributes(ctx api.Context, options *api.ListOptions) (*rbac.ProtectedAttributeList, error)
	CreateProtectedAttribute(ctx api.Context, pa *rbac.ProtectedAttribute) error
	UpdateProtectedAttribute(ctx api.Context, pa *rbac.ProtectedAttribute) error
	GetProtectedAttribute(ctx api.Context, name string) (*rbac.ProtectedAttribute, error)
	DeleteProtectedAttribute(ctx api.Context, name string) error
	WatchProtectedAttributes(ctx api.Context, options *api.ListOptions) (watch.Interface, error)
}

type storage struct {
	rest.StandardStorage
}

// NewRegistry returns a new Registry of ProtectedAttribute objects.
func NewRegistry(s rest.StandardStorage) Registry {
	return &storage{s}
}

func (s *storage) ListProtectedAttributes(ctx api.Context, options *api.ListOptions) (*rbac.ProtectedAttributeList, error) {
	obj, err := s.List(ctx, options)
	if err != nil {
		return nil, err
	}
	return obj.(*rbac.ProtectedAttributeList), nil
}

func (s *storage) CreateProtectedAttribute(ctx api.Context, pa *rbac.ProtectedAttribute) error {
	_, err := s.Create(ctx, pa)
	return err
}

func (s *storage) UpdateProtectedAttribute(ctx api.Context, pa *rbac.ProtectedAttribute) error {
	_, _, err := s.Update(ctx, pa.Name, rest.DefaultUpdatedObjectInfo(pa, api.Scheme))
	return err
}

func (s *storage) GetProtectedAttribute(ctx api.Context, name string) (*rbac.ProtectedAttribute, error) {
	obj, err := s.Get(ctx, name)
	if err != nil {
		return nil, err
	}
	return obj.(*rbac.ProtectedAttribute), nil
}

func (s *storage) DeleteProtectedAttribute(ctx api.Context, name string) error {
	_, err := s.Delete(ctx, name, nil)
	return err
}

func (s *storage) WatchProtectedAttributes(ctx api.Context, options *api.ListOptions) (watch.Interface, error) {
	return s.Watch(ctx, options)
}
