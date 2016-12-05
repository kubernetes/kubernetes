/*
Copyright 2016 The Kubernetes Authors.

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

package clusterrole

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/watch"
)

// Registry is an interface for things that know how to store ClusterRoles.
type Registry interface {
	ListClusterRoles(ctx api.Context, options *api.ListOptions) (*rbac.ClusterRoleList, error)
	CreateClusterRole(ctx api.Context, clusterRole *rbac.ClusterRole) error
	UpdateClusterRole(ctx api.Context, clusterRole *rbac.ClusterRole) error
	GetClusterRole(ctx api.Context, name string, options *metav1.GetOptions) (*rbac.ClusterRole, error)
	DeleteClusterRole(ctx api.Context, name string) error
	WatchClusterRoles(ctx api.Context, options *api.ListOptions) (watch.Interface, error)
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

func (s *storage) ListClusterRoles(ctx api.Context, options *api.ListOptions) (*rbac.ClusterRoleList, error) {
	obj, err := s.List(ctx, options)
	if err != nil {
		return nil, err
	}

	return obj.(*rbac.ClusterRoleList), nil
}

func (s *storage) CreateClusterRole(ctx api.Context, clusterRole *rbac.ClusterRole) error {
	_, err := s.Create(ctx, clusterRole)
	return err
}

func (s *storage) UpdateClusterRole(ctx api.Context, clusterRole *rbac.ClusterRole) error {
	_, _, err := s.Update(ctx, clusterRole.Name, rest.DefaultUpdatedObjectInfo(clusterRole, api.Scheme))
	return err
}

func (s *storage) WatchClusterRoles(ctx api.Context, options *api.ListOptions) (watch.Interface, error) {
	return s.Watch(ctx, options)
}

func (s *storage) GetClusterRole(ctx api.Context, name string, options *metav1.GetOptions) (*rbac.ClusterRole, error) {
	obj, err := s.Get(ctx, name, options)
	if err != nil {
		return nil, err
	}
	return obj.(*rbac.ClusterRole), nil
}

func (s *storage) DeleteClusterRole(ctx api.Context, name string) error {
	_, err := s.Delete(ctx, name, nil)
	return err
}

// AuthorizerAdapter adapts the registry to the authorizer interface
type AuthorizerAdapter struct {
	Registry Registry
}

func (a AuthorizerAdapter) GetClusterRole(name string) (*rbac.ClusterRole, error) {
	return a.Registry.GetClusterRole(api.NewContext(), name, &metav1.GetOptions{})
}
