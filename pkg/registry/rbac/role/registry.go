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
	"context"

	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/kubernetes/pkg/apis/rbac"
	rbacv1helpers "k8s.io/kubernetes/pkg/apis/rbac/v1"
)

// Registry is an interface for things that know how to store Roles.
type Registry interface {
	GetRole(ctx context.Context, name string, options *metav1.GetOptions) (*rbacv1.Role, error)
}

// storage puts strong typing around storage calls
type storage struct {
	rest.Getter
}

// NewRegistry returns a new Registry interface for the given Storage. Any mismatched
// types will panic.
func NewRegistry(s rest.StandardStorage) Registry {
	return &storage{s}
}

func (s *storage) GetRole(ctx context.Context, name string, options *metav1.GetOptions) (*rbacv1.Role, error) {
	obj, err := s.Get(ctx, name, options)
	if err != nil {
		return nil, err
	}

	ret := &rbacv1.Role{}
	if err := rbacv1helpers.Convert_rbac_Role_To_v1_Role(obj.(*rbac.Role), ret, nil); err != nil {
		return nil, err
	}
	return ret, nil
}

// AuthorizerAdapter adapts the registry to the authorizer interface
type AuthorizerAdapter struct {
	Registry Registry
}

// GetRole returns the corresponding Role by name in specified namespace
func (a AuthorizerAdapter) GetRole(namespace, name string) (*rbacv1.Role, error) {
	return a.Registry.GetRole(genericapirequest.WithNamespace(genericapirequest.NewContext(), namespace), name, &metav1.GetOptions{})
}
