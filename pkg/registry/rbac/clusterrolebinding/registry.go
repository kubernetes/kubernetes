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

package clusterrolebinding

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/watch"
)

// Registry is an interface for things that know how to store ClusterRoleBindings.
type Registry interface {
	ListClusterRoleBindings(ctx api.Context, options *api.ListOptions) (*rbac.ClusterRoleBindingList, error)
	CreateClusterRoleBinding(ctx api.Context, clusterRoleBinding *rbac.ClusterRoleBinding) error
	UpdateClusterRoleBinding(ctx api.Context, clusterRoleBinding *rbac.ClusterRoleBinding) error
	GetClusterRoleBinding(ctx api.Context, name string) (*rbac.ClusterRoleBinding, error)
	DeleteClusterRoleBinding(ctx api.Context, name string) error
	WatchClusterRoleBindings(ctx api.Context, options *api.ListOptions) (watch.Interface, error)
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

func (s *storage) ListClusterRoleBindings(ctx api.Context, options *api.ListOptions) (*rbac.ClusterRoleBindingList, error) {
	obj, err := s.List(ctx, options)
	if err != nil {
		return nil, err
	}

	return obj.(*rbac.ClusterRoleBindingList), nil
}

func (s *storage) CreateClusterRoleBinding(ctx api.Context, clusterRoleBinding *rbac.ClusterRoleBinding) error {
	_, err := s.Create(ctx, clusterRoleBinding)
	return err
}

func (s *storage) UpdateClusterRoleBinding(ctx api.Context, clusterRoleBinding *rbac.ClusterRoleBinding) error {
	_, _, err := s.Update(ctx, clusterRoleBinding.Name, rest.DefaultUpdatedObjectInfo(clusterRoleBinding, api.Scheme))
	return err
}

func (s *storage) WatchClusterRoleBindings(ctx api.Context, options *api.ListOptions) (watch.Interface, error) {
	return s.Watch(ctx, options)
}

func (s *storage) GetClusterRoleBinding(ctx api.Context, name string) (*rbac.ClusterRoleBinding, error) {
	obj, err := s.Get(ctx, name)
	if err != nil {
		return nil, err
	}
	return obj.(*rbac.ClusterRoleBinding), nil
}

func (s *storage) DeleteClusterRoleBinding(ctx api.Context, name string) error {
	_, err := s.Delete(ctx, name, nil)
	return err
}

// AuthorizerAdapter adapts the registry to the authorizer interface
type AuthorizerAdapter struct {
	Registry Registry
}

func (a AuthorizerAdapter) ListClusterRoleBindings() ([]*rbac.ClusterRoleBinding, error) {
	list, err := a.Registry.ListClusterRoleBindings(api.NewContext(), &api.ListOptions{})
	if err != nil {
		return nil, err
	}

	ret := []*rbac.ClusterRoleBinding{}
	for i := range list.Items {
		ret = append(ret, &list.Items[i])
	}
	return ret, nil
}
