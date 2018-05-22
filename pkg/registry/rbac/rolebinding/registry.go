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
	"context"

	rbacv1 "k8s.io/api/rbac/v1"
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/kubernetes/pkg/apis/rbac"
	rbacv1helpers "k8s.io/kubernetes/pkg/apis/rbac/v1"
)

// Registry is an interface for things that know how to store RoleBindings.
type Registry interface {
	ListRoleBindings(ctx context.Context, options *metainternalversion.ListOptions) (*rbacv1.RoleBindingList, error)
}

// storage puts strong typing around storage calls
type storage struct {
	rest.Lister
}

// NewRegistry returns a new Registry interface for the given Storage. Any mismatched
// types will panic.
func NewRegistry(s rest.StandardStorage) Registry {
	return &storage{s}
}

func (s *storage) ListRoleBindings(ctx context.Context, options *metainternalversion.ListOptions) (*rbacv1.RoleBindingList, error) {
	obj, err := s.List(ctx, options)
	if err != nil {
		return nil, err
	}

	ret := &rbacv1.RoleBindingList{}
	if err := rbacv1helpers.Convert_rbac_RoleBindingList_To_v1_RoleBindingList(obj.(*rbac.RoleBindingList), ret, nil); err != nil {
		return nil, err
	}
	return ret, nil
}

// AuthorizerAdapter adapts the registry to the authorizer interface
type AuthorizerAdapter struct {
	Registry Registry
}

func (a AuthorizerAdapter) ListRoleBindings(namespace string) ([]*rbacv1.RoleBinding, error) {
	list, err := a.Registry.ListRoleBindings(genericapirequest.WithNamespace(genericapirequest.NewContext(), namespace), &metainternalversion.ListOptions{})
	if err != nil {
		return nil, err
	}

	ret := []*rbacv1.RoleBinding{}
	for i := range list.Items {
		ret = append(ret, &list.Items[i])
	}
	return ret, nil
}
