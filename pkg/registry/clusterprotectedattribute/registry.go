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

package clusterprotectedattribute

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/watch"
)

// Registry interface exposes operations on ClusterProtectedAttribute's.
type Registry interface {
	ListClusterProtectedAttributes(ctx api.Context, options *api.ListOptions) (*rbac.ClusterProtectedAttributeList, error)
	CreateClusterProtectedAttribute(ctx api.Context, cpa *rbac.ClusterProtectedAttribute) error
	UpdateClusterProtectedAttribute(ctx api.Context, cpa *rbac.ClusterProtectedAttribute) error
	GetClusterProtectedAttribute(ctx api.Context, name string) (*rbac.ClusterProtectedAttribute, error)
	DeleteClusterProtectedAttribute(ctx api.Context, name string) error
	WatchClusterProtectedAttributes(ctx api.Context, options *api.ListOptions) (watch.Interface, error)
}

type storage struct {
	rest.StandardStorage
}

// NewRegistry returns a new Registry of ClusterProtectedAttribute
// objects.
func NewRegistry(s rest.StandardStorage) Registry {
	return &storage{s}
}

func (s *storage) ListClusterProtectedAttributes(ctx api.Context, options *api.ListOptions) (*rbac.ClusterProtectedAttributeList, error) {
	obj, err := s.List(ctx, options)
	if err != nil {
		return nil, err
	}
	return obj.(*rbac.ClusterProtectedAttributeList), nil
}

func (s *storage) CreateClusterProtectedAttribute(ctx api.Context, cpa *rbac.ClusterProtectedAttribute) error {
	_, err := s.Create(ctx, cpa)
	return err
}

func (s *storage) UpdateClusterProtectedAttribute(ctx api.Context, cpa *rbac.ClusterProtectedAttribute) error {
	_, _, err := s.Update(ctx, cpa.Name, rest.DefaultUpdatedObjectInfo(cpa, api.Scheme))
	return err
}

func (s *storage) GetClusterProtectedAttribute(ctx api.Context, name string) (*rbac.ClusterProtectedAttribute, error) {
	obj, err := s.Get(ctx, name)
	if err != nil {
		return nil, err
	}
	return obj.(*rbac.ClusterProtectedAttribute), nil
}

func (s *storage) DeleteClusterProtectedAttribute(ctx api.Context, name string) error {
	_, err := s.Delete(ctx, name, nil)
	return err
}

func (s *storage) WatchClusterProtectedAttributes(ctx api.Context, options *api.ListOptions) (watch.Interface, error) {
	return s.Watch(ctx, options)
}
