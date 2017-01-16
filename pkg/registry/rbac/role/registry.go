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

package role

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/watch"
	genericapirequest "k8s.io/apiserver/pkg/request"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/apis/rbac"
)

// Registry is an interface for things that know how to store Roles.
type Registry interface {
	ListRoles(ctx genericapirequest.Context, options *api.ListOptions) (*rbac.RoleList, error)
	CreateRole(ctx genericapirequest.Context, role *rbac.Role) error
	UpdateRole(ctx genericapirequest.Context, role *rbac.Role) error
	GetRole(ctx genericapirequest.Context, name string, options *metav1.GetOptions) (*rbac.Role, error)
	DeleteRole(ctx genericapirequest.Context, name string) error
	WatchRoles(ctx genericapirequest.Context, options *api.ListOptions) (watch.Interface, error)
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

func (s *storage) ListRoles(ctx genericapirequest.Context, options *api.ListOptions) (*rbac.RoleList, error) {
	obj, err := s.List(ctx, options)
	if err != nil {
		return nil, err
	}

	return obj.(*rbac.RoleList), nil
}

func (s *storage) CreateRole(ctx genericapirequest.Context, role *rbac.Role) error {
	_, err := s.Create(ctx, role)
	return err
}

func (s *storage) UpdateRole(ctx genericapirequest.Context, role *rbac.Role) error {
	_, _, err := s.Update(ctx, role.Name, rest.DefaultUpdatedObjectInfo(role, api.Scheme))
	return err
}

func (s *storage) WatchRoles(ctx genericapirequest.Context, options *api.ListOptions) (watch.Interface, error) {
	return s.Watch(ctx, options)
}

func (s *storage) GetRole(ctx genericapirequest.Context, name string, options *metav1.GetOptions) (*rbac.Role, error) {
	obj, err := s.Get(ctx, name, options)
	if err != nil {
		return nil, err
	}
	return obj.(*rbac.Role), nil
}

func (s *storage) DeleteRole(ctx genericapirequest.Context, name string) error {
	_, err := s.Delete(ctx, name, nil)
	return err
}

// AuthorizerAdapter adapts the registry to the authorizer interface
type AuthorizerAdapter struct {
	Registry Registry
}

func (a AuthorizerAdapter) GetRole(namespace, name string) (*rbac.Role, error) {
	return a.Registry.GetRole(genericapirequest.WithNamespace(genericapirequest.NewContext(), namespace), name, &metav1.GetOptions{})
}
