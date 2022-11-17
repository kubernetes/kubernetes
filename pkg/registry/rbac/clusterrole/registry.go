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
	"context"

	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/kubernetes/pkg/apis/rbac"
	rbacv1helpers "k8s.io/kubernetes/pkg/apis/rbac/v1"
)

// Registry is an interface for things that know how to store ClusterRoles.
type Registry interface {
	GetClusterRole(ctx context.Context, name string, options *metav1.GetOptions) (*rbacv1.ClusterRole, error)
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

func (s *storage) GetClusterRole(ctx context.Context, name string, options *metav1.GetOptions) (*rbacv1.ClusterRole, error) {
	obj, err := s.Get(ctx, name, options)
	if err != nil {
		return nil, err
	}

	ret := &rbacv1.ClusterRole{}
	if err := rbacv1helpers.Convert_rbac_ClusterRole_To_v1_ClusterRole(obj.(*rbac.ClusterRole), ret, nil); err != nil {
		return nil, err
	}
	return ret, nil
}

// AuthorizerAdapter adapts the registry to the authorizer interface
type AuthorizerAdapter struct {
	Registry Registry
}

// GetClusterRole returns the corresponding ClusterRole by name
func (a AuthorizerAdapter) GetClusterRole(ctx context.Context, name string) (*rbacv1.ClusterRole, error) {
	return a.Registry.GetClusterRole(ctx, name, &metav1.GetOptions{})
}
