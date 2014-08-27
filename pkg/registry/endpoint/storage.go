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

package endpoint

import (
	"errors"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// Storage adapts endpoints into apiserver's RESTStorage model.
type Storage struct {
	registry Registry
}

// NewStorage returns a new Storage implementation for endpoints
func NewStorage(registry Registry) apiserver.RESTStorage {
	return &Storage{
		registry: registry,
	}
}

// Get satisfies the RESTStorage interface but is unimplemented
func (rs *Storage) Get(id string) (interface{}, error) {
	return rs.registry.GetEndpoints(id)
}

// List satisfies the RESTStorage interface but is unimplemented
func (rs *Storage) List(selector labels.Selector) (interface{}, error) {
	return nil, errors.New("unimplemented")
}

// Watch returns Endpoint events via a watch.Interface.
// It implements apiserver.ResourceWatcher.
func (rs *Storage) Watch(label, field labels.Selector, resourceVersion uint64) (watch.Interface, error) {
	return rs.registry.WatchEndpoints(label, field, resourceVersion)
}

// Create satisfies the RESTStorage interface but is unimplemented
func (rs *Storage) Create(obj interface{}) (<-chan interface{}, error) {
	return nil, errors.New("unimplemented")
}

// Update satisfies the RESTStorage interface but is unimplemented
func (rs *Storage) Update(obj interface{}) (<-chan interface{}, error) {
	return nil, errors.New("unimplemented")
}

// Delete satisfies the RESTStorage interface but is unimplemented
func (rs *Storage) Delete(id string) (<-chan interface{}, error) {
	return nil, errors.New("unimplemented")
}

// New implements the RESTStorage interface
func (rs Storage) New() interface{} {
	return &api.Endpoints{}
}
