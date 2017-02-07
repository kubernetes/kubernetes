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

package node

import (
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/watch"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/kubernetes/pkg/api"
)

// Registry is an interface for things that know how to store node.
type Registry interface {
	ListNodes(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (*api.NodeList, error)
	CreateNode(ctx genericapirequest.Context, node *api.Node) error
	UpdateNode(ctx genericapirequest.Context, node *api.Node) error
	GetNode(ctx genericapirequest.Context, nodeID string, options *metav1.GetOptions) (*api.Node, error)
	DeleteNode(ctx genericapirequest.Context, nodeID string) error
	WatchNodes(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (watch.Interface, error)
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

func (s *storage) ListNodes(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (*api.NodeList, error) {
	obj, err := s.List(ctx, options)
	if err != nil {
		return nil, err
	}

	return obj.(*api.NodeList), nil
}

func (s *storage) CreateNode(ctx genericapirequest.Context, node *api.Node) error {
	_, err := s.Create(ctx, node)
	return err
}

func (s *storage) UpdateNode(ctx genericapirequest.Context, node *api.Node) error {
	_, _, err := s.Update(ctx, node.Name, rest.DefaultUpdatedObjectInfo(node, api.Scheme))
	return err
}

func (s *storage) WatchNodes(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (watch.Interface, error) {
	return s.Watch(ctx, options)
}

func (s *storage) GetNode(ctx genericapirequest.Context, name string, options *metav1.GetOptions) (*api.Node, error) {
	obj, err := s.Get(ctx, name, options)
	if err != nil {
		return nil, err
	}
	return obj.(*api.Node), nil
}

func (s *storage) DeleteNode(ctx genericapirequest.Context, name string) error {
	_, _, err := s.Delete(ctx, name, nil)
	return err
}
