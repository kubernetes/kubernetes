/*
Copyright 2014 Google Inc. All rights reserved.

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

package resourcequota

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/rest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// Registry is an interface implemented by things that know how to store ResourceQuota objects.
type Registry interface {
	// ListResourceQuotas obtains a list of pods having labels which match selector.
	ListResourceQuotas(ctx api.Context, selector labels.Selector) (*api.ResourceQuotaList, error)
	// Watch for new/changed/deleted pods
	WatchResourceQuotas(ctx api.Context, label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error)
	// Get a specific pod
	GetResourceQuota(ctx api.Context, podID string) (*api.ResourceQuota, error)
	// Create a pod based on a specification.
	CreateResourceQuota(ctx api.Context, pod *api.ResourceQuota) error
	// Update an existing pod
	UpdateResourceQuota(ctx api.Context, pod *api.ResourceQuota) error
	// Delete an existing pod
	DeleteResourceQuota(ctx api.Context, podID string) error
}

// Storage is an interface for a standard REST Storage backend
// TODO: move me somewhere common
type Storage interface {
	rest.GracefulDeleter
	rest.Lister
	rest.Getter
	rest.Watcher

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

func (s *storage) ListResourceQuotas(ctx api.Context, label labels.Selector) (*api.ResourceQuotaList, error) {
	obj, err := s.List(ctx, label, fields.Everything())
	if err != nil {
		return nil, err
	}
	return obj.(*api.ResourceQuotaList), nil
}

func (s *storage) WatchResourceQuotas(ctx api.Context, label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return s.Watch(ctx, label, field, resourceVersion)
}

func (s *storage) GetResourceQuota(ctx api.Context, podID string) (*api.ResourceQuota, error) {
	obj, err := s.Get(ctx, podID)
	if err != nil {
		return nil, err
	}
	return obj.(*api.ResourceQuota), nil
}

func (s *storage) CreateResourceQuota(ctx api.Context, pod *api.ResourceQuota) error {
	_, err := s.Create(ctx, pod)
	return err
}

func (s *storage) UpdateResourceQuota(ctx api.Context, pod *api.ResourceQuota) error {
	_, _, err := s.Update(ctx, pod)
	return err
}

func (s *storage) DeleteResourceQuota(ctx api.Context, podID string) error {
	_, err := s.Delete(ctx, podID, nil)
	return err
}
