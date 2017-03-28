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

package rolebinding

import (
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/watch"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/rbac"
)

// Registry is an interface for things that know how to store RoleBindings.
type Registry interface {
	ListRoleBindings(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (*rbac.RoleBindingList, error)
	CreateRoleBinding(ctx genericapirequest.Context, roleBinding *rbac.RoleBinding) error
	UpdateRoleBinding(ctx genericapirequest.Context, roleBinding *rbac.RoleBinding) error
	GetRoleBinding(ctx genericapirequest.Context, name string, options *metav1.GetOptions) (*rbac.RoleBinding, error)
	DeleteRoleBinding(ctx genericapirequest.Context, name string) error
	WatchRoleBindings(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (watch.Interface, error)
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

func (s *storage) ListRoleBindings(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (*rbac.RoleBindingList, error) {
	obj, err := s.List(ctx, options)
	if err != nil {
		return nil, err
	}

	return obj.(*rbac.RoleBindingList), nil
}

func (s *storage) CreateRoleBinding(ctx genericapirequest.Context, roleBinding *rbac.RoleBinding) error {
	// TODO(ericchiang): add additional validation
	_, err := s.Create(ctx, roleBinding)
	return err
}

func (s *storage) UpdateRoleBinding(ctx genericapirequest.Context, roleBinding *rbac.RoleBinding) error {
	_, _, err := s.Update(ctx, roleBinding.Name, rest.DefaultUpdatedObjectInfo(roleBinding, api.Scheme))
	return err
}

func (s *storage) WatchRoleBindings(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (watch.Interface, error) {
	return s.Watch(ctx, options)
}

func (s *storage) GetRoleBinding(ctx genericapirequest.Context, name string, options *metav1.GetOptions) (*rbac.RoleBinding, error) {
	obj, err := s.Get(ctx, name, options)
	if err != nil {
		return nil, err
	}
	return obj.(*rbac.RoleBinding), nil
}

func (s *storage) DeleteRoleBinding(ctx genericapirequest.Context, name string) error {
	_, _, err := s.Delete(ctx, name, nil)
	return err
}

// AuthorizerAdapter adapts the registry to the authorizer interface
type AuthorizerAdapter struct {
	Registry Registry
}

func (a AuthorizerAdapter) ListRoleBindings(namespace string) ([]*rbac.RoleBinding, error) {
	list, err := a.Registry.ListRoleBindings(genericapirequest.WithNamespace(genericapirequest.NewContext(), namespace), &metainternalversion.ListOptions{})
	if err != nil {
		return nil, err
	}

	ret := []*rbac.RoleBinding{}
	for i := range list.Items {
		ret = append(ret, &list.Items[i])
	}
	return ret, nil
}
