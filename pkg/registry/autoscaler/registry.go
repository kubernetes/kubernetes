/*
Copyright 2015 Google Inc. All rights reserved.

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

package autoscaler

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// registry implements custom changes to generic.Etcd.
type Registry interface {
	// ListAutoScalers obtains a list of autoScalers having labels which match selector.
	ListAutoScalers(ctx api.Context, selector labels.Selector) (*api.AutoScalerList, error)
	// Watch for new/changed/deleted autoScalers
	WatchAutoScalers(ctx api.Context, label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error)
	// Get a specific autoScaler
	GetAutoScaler(ctx api.Context, autoScalerID string) (*api.AutoScaler, error)
	// Create a autoScaler based on a specification.
	CreateAutoScaler(ctx api.Context, autoScaler *api.AutoScaler) error
	// Update an existing autoScaler
	UpdateAutoScaler(ctx api.Context, autoScaler *api.AutoScaler) error
	// Delete an existing autoScaler
	DeleteAutoScaler(ctx api.Context, autoScalerID string) error
}

// Storage is an interface for a standard REST Storage backend
// TODO: move me somewhere common
type Storage interface {
	apiserver.RESTDeleter
	apiserver.RESTLister
	apiserver.RESTGetter
	apiserver.ResourceWatcher

	Create(ctx api.Context, obj runtime.Object) (runtime.Object, error)
	Update(ctx api.Context, obj runtime.Object) (runtime.Object, bool, error)
}

// storage puts strong typing around storage calls
type storage struct {
	Storage
}

// NewRegistry returns a new Registry interface for the given Storage. Any mismatched
// types will panic.
func NewRegistry(s Storage) Registry {
	return &storage{s}
}

func (s *storage) ListAutoScalers(ctx api.Context, label labels.Selector) (*api.AutoScalerList, error) {
	obj, err := s.List(ctx, label, fields.Everything())
	if err != nil {
		return nil, err
	}
	return obj.(*api.AutoScalerList), nil
}

func (s *storage) WatchAutoScalers(ctx api.Context, label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return s.Watch(ctx, label, field, resourceVersion)
}

func (s *storage) GetAutoScaler(ctx api.Context, id string) (*api.AutoScaler, error) {
	obj, err := s.Get(ctx, id)
	if err != nil {
		return nil, err
	}
	return obj.(*api.AutoScaler), nil
}

func (s *storage) CreateAutoScaler(ctx api.Context, autoScaler *api.AutoScaler) error {
	_, err := s.Create(ctx, autoScaler)
	return err
}

func (s *storage) UpdateAutoScaler(ctx api.Context, autoScaler *api.AutoScaler) error {
	_, _, err := s.Update(ctx, autoScaler)
	return err
}

func (s *storage) DeleteAutoScaler(ctx api.Context, id string) error {
	_, err := s.Delete(ctx, id)
	return err
}
