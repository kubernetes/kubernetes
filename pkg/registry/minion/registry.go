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

package minion

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"
)

// Registry is an interface for things that know how to store node.
type Registry interface {
	ListMinions(ctx api.Context, label labels.Selector, field fields.Selector) (*api.NodeList, error)
	CreateMinion(ctx api.Context, minion *api.Node) error
	UpdateMinion(ctx api.Context, minion *api.Node) error
	GetMinion(ctx api.Context, minionID string) (*api.Node, error)
	DeleteMinion(ctx api.Context, minionID string) error
	WatchMinions(ctx api.Context, label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error)
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

func (s *storage) ListMinions(ctx api.Context, label labels.Selector, field fields.Selector) (*api.NodeList, error) {
	obj, err := s.List(ctx, label, field)
	if err != nil {
		return nil, err
	}

	return obj.(*api.NodeList), nil
}

func (s *storage) CreateMinion(ctx api.Context, node *api.Node) error {
	_, err := s.Create(ctx, node)
	return err
}

func (s *storage) UpdateMinion(ctx api.Context, node *api.Node) error {
	_, _, err := s.Update(ctx, node)
	return err
}

func (s *storage) WatchMinions(ctx api.Context, label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return s.Watch(ctx, label, field, resourceVersion)
}

func (s *storage) GetMinion(ctx api.Context, name string) (*api.Node, error) {
	obj, err := s.Get(ctx, name)
	if err != nil {
		return nil, err
	}
	return obj.(*api.Node), nil
}

func (s *storage) DeleteMinion(ctx api.Context, name string) error {
	_, err := s.Delete(ctx, name, nil)
	return err
}
