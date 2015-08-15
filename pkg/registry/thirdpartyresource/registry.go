/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package thirdpartyresource

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/expapi"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"
)

// Registry is an interface implemented by things that know how to store ThirdPartyResource objects.
type Registry interface {
	// ListThirdPartyResources obtains a list of ThirdPartyResources having labels which match selector.
	ListThirdPartyResources(ctx api.Context, selector labels.Selector) (*expapi.ThirdPartyResourceList, error)
	// Watch for new/changed/deleted ThirdPartyResources
	WatchThirdPartyResources(ctx api.Context, label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error)
	// Get a specific ThirdPartyResource
	GetThirdPartyResource(ctx api.Context, name string) (*expapi.ThirdPartyResource, error)
	// Create a ThirdPartyResource based on a specification.
	CreateThirdPartyResource(ctx api.Context, resource *expapi.ThirdPartyResource) (*expapi.ThirdPartyResource, error)
	// Update an existing ThirdPartyResource
	UpdateThirdPartyResource(ctx api.Context, resource *expapi.ThirdPartyResource) (*expapi.ThirdPartyResource, error)
	// Delete an existing ThirdPartyResource
	DeleteThirdPartyResource(ctx api.Context, name string) error
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

func (s *storage) ListThirdPartyResources(ctx api.Context, label labels.Selector) (*expapi.ThirdPartyResourceList, error) {
	obj, err := s.List(ctx, label, fields.Everything())
	if err != nil {
		return nil, err
	}
	return obj.(*expapi.ThirdPartyResourceList), nil
}

func (s *storage) WatchThirdPartyResources(ctx api.Context, label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return s.Watch(ctx, label, field, resourceVersion)
}

func (s *storage) GetThirdPartyResource(ctx api.Context, name string) (*expapi.ThirdPartyResource, error) {
	obj, err := s.Get(ctx, name)
	if err != nil {
		return nil, err
	}
	return obj.(*expapi.ThirdPartyResource), nil
}

func (s *storage) CreateThirdPartyResource(ctx api.Context, ThirdPartyResource *expapi.ThirdPartyResource) (*expapi.ThirdPartyResource, error) {
	obj, err := s.Create(ctx, ThirdPartyResource)
	return obj.(*expapi.ThirdPartyResource), err
}

func (s *storage) UpdateThirdPartyResource(ctx api.Context, ThirdPartyResource *expapi.ThirdPartyResource) (*expapi.ThirdPartyResource, error) {
	obj, _, err := s.Update(ctx, ThirdPartyResource)
	return obj.(*expapi.ThirdPartyResource), err
}

func (s *storage) DeleteThirdPartyResource(ctx api.Context, name string) error {
	_, err := s.Delete(ctx, name, nil)
	return err
}
