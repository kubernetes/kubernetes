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

package resourcequota

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"
)

// Registry is an interface implemented by things that know how to store ResourceQuota objects.
type Registry interface {
	// ListResourceQuotas obtains a list of resourceQuotas having labels which match selector.
	ListResourceQuotas(ctx api.Context, selector labels.Selector) (*api.ResourceQuotaList, error)
	// Watch for new/changed/deleted resourceQuotas
	WatchResourceQuotas(ctx api.Context, label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error)
	// Get a specific resourceQuota
	GetResourceQuota(ctx api.Context, resourceQuotaID string) (*api.ResourceQuota, error)
	// Create a resourceQuota based on a specification.
	CreateResourceQuota(ctx api.Context, resourceQuota *api.ResourceQuota) error
	// Update an existing resourceQuota
	UpdateResourceQuota(ctx api.Context, resourceQuota *api.ResourceQuota) error
	// Delete an existing resourceQuota
	DeleteResourceQuota(ctx api.Context, resourceQuotaID string) error
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

func (s *storage) GetResourceQuota(ctx api.Context, resourceQuotaID string) (*api.ResourceQuota, error) {
	obj, err := s.Get(ctx, resourceQuotaID)
	if err != nil {
		return nil, err
	}
	return obj.(*api.ResourceQuota), nil
}

func (s *storage) CreateResourceQuota(ctx api.Context, resourceQuota *api.ResourceQuota) error {
	_, err := s.Create(ctx, resourceQuota)
	return err
}

func (s *storage) UpdateResourceQuota(ctx api.Context, resourceQuota *api.ResourceQuota) error {
	_, _, err := s.Update(ctx, resourceQuota)
	return err
}

func (s *storage) DeleteResourceQuota(ctx api.Context, resourceQuotaID string) error {
	_, err := s.Delete(ctx, resourceQuotaID, nil)
	return err
}
