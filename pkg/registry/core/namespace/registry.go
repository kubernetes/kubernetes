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

package namespace

import (
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/watch"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/kubernetes/pkg/api"
)

// Registry is an interface implemented by things that know how to store Namespace objects.
type Registry interface {
	ListNamespaces(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (*api.NamespaceList, error)
	WatchNamespaces(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (watch.Interface, error)
	GetNamespace(ctx genericapirequest.Context, namespaceID string, options *metav1.GetOptions) (*api.Namespace, error)
	CreateNamespace(ctx genericapirequest.Context, namespace *api.Namespace) error
	UpdateNamespace(ctx genericapirequest.Context, namespace *api.Namespace) error
	DeleteNamespace(ctx genericapirequest.Context, namespaceID string) error
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

func (s *storage) ListNamespaces(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (*api.NamespaceList, error) {
	obj, err := s.List(ctx, options)
	if err != nil {
		return nil, err
	}
	return obj.(*api.NamespaceList), nil
}

func (s *storage) WatchNamespaces(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (watch.Interface, error) {
	return s.Watch(ctx, options)
}

func (s *storage) GetNamespace(ctx genericapirequest.Context, namespaceName string, options *metav1.GetOptions) (*api.Namespace, error) {
	obj, err := s.Get(ctx, namespaceName, options)
	if err != nil {
		return nil, err
	}
	return obj.(*api.Namespace), nil
}

func (s *storage) CreateNamespace(ctx genericapirequest.Context, namespace *api.Namespace) error {
	_, err := s.Create(ctx, namespace)
	return err
}

func (s *storage) UpdateNamespace(ctx genericapirequest.Context, namespace *api.Namespace) error {
	_, _, err := s.Update(ctx, namespace.Name, rest.DefaultUpdatedObjectInfo(namespace, api.Scheme))
	return err
}

func (s *storage) DeleteNamespace(ctx genericapirequest.Context, namespaceID string) error {
	_, err := s.Delete(ctx, namespaceID, nil)
	return err
}
