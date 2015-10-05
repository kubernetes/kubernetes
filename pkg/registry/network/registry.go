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

package network

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"
)

// Registry is an interface implemented by things that know how to store Network objects.
type Registry interface {
	// ListNetworks obtains a list of Networks having labels which match selector.
	ListNetworks(ctx api.Context, selector labels.Selector) (*api.NetworkList, error)
	// Watch for new/changed/deleted Networks
	WatchNetworks(ctx api.Context, label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error)
	// Get a specific Network
	GetNetwork(ctx api.Context, NetworkID string) (*api.Network, error)
	// Create a Network based on a specification.
	CreateNetwork(ctx api.Context, Network *api.Network) error
	// Update an existing Network
	UpdateNetwork(ctx api.Context, Network *api.Network) error
	// Delete an existing Network
	DeleteNetwork(ctx api.Context, NetworkID string) error
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

func (s *storage) ListNetworks(ctx api.Context, label labels.Selector) (*api.NetworkList, error) {
	obj, err := s.List(ctx, label, fields.Everything())
	if err != nil {
		return nil, err
	}
	return obj.(*api.NetworkList), nil
}

func (s *storage) WatchNetworks(ctx api.Context, label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return s.Watch(ctx, label, field, resourceVersion)
}

func (s *storage) GetNetwork(ctx api.Context, NetworkName string) (*api.Network, error) {
	obj, err := s.Get(ctx, NetworkName)
	if err != nil {
		return nil, err
	}
	return obj.(*api.Network), nil
}

func (s *storage) CreateNetwork(ctx api.Context, Network *api.Network) error {
	_, err := s.Create(ctx, Network)
	return err
}

func (s *storage) UpdateNetwork(ctx api.Context, Network *api.Network) error {
	_, _, err := s.Update(ctx, Network)
	return err
}

func (s *storage) DeleteNetwork(ctx api.Context, NetworkID string) error {
	_, err := s.Delete(ctx, NetworkID, nil)
	return err
}
