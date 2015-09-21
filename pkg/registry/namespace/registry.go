/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"
)

// Registry is an interface implemented by things that know how to store Namespace objects.
type Registry interface {
	// ListNamespaces obtains a list of namespaces having labels which match selector.
	ListNamespaces(ctx api.Context, selector labels.Selector) (*api.NamespaceList, error)
	// Watch for new/changed/deleted namespaces
	WatchNamespaces(ctx api.Context, label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error)
	// Get a specific namespace
	GetNamespace(ctx api.Context, namespaceID string) (*api.Namespace, error)
	// Create a namespace based on a specification.
	CreateNamespace(ctx api.Context, namespace *api.Namespace) error
	// Update an existing namespace
	UpdateNamespace(ctx api.Context, namespace *api.Namespace) error
	// Delete an existing namespace
	DeleteNamespace(ctx api.Context, namespaceID string) error
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

func (s *storage) ListNamespaces(ctx api.Context, label labels.Selector) (*api.NamespaceList, error) {
	obj, err := s.List(ctx, label, fields.Everything())
	if err != nil {
		return nil, err
	}
	return obj.(*api.NamespaceList), nil
}

func (s *storage) WatchNamespaces(ctx api.Context, label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return s.Watch(ctx, label, field, resourceVersion)
}

func (s *storage) GetNamespace(ctx api.Context, namespaceName string) (*api.Namespace, error) {
	obj, err := s.Get(ctx, namespaceName)
	if err != nil {
		return nil, err
	}
	return obj.(*api.Namespace), nil
}

func (s *storage) CreateNamespace(ctx api.Context, namespace *api.Namespace) error {
	_, err := s.Create(ctx, namespace)
	return err
}

func (s *storage) UpdateNamespace(ctx api.Context, namespace *api.Namespace) error {
	_, _, err := s.Update(ctx, namespace)
	return err
}

func (s *storage) DeleteNamespace(ctx api.Context, namespaceID string) error {
	_, err := s.Delete(ctx, namespaceID, nil)
	return err
}
