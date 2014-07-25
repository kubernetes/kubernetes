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

package registry

import (
	"fmt"

	"code.google.com/p/go-uuid/uuid"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
)

// BuildRegistryStorage is an implementation of RESTStorage for the api server.
type BuildRegistryStorage struct {
	registry BuildRegistry
}

func NewBuildRegistryStorage(registry BuildRegistry) apiserver.RESTStorage {
	return &BuildRegistryStorage{
		registry: registry,
	}
}

// List obtains a list of Builds that match selector.
func (storage *BuildRegistryStorage) List(selector labels.Selector) (interface{}, error) {
	result := api.BuildList{}
	builds, err := storage.registry.ListBuilds()
	if err == nil {
		for _, build := range builds.Items {
			result.Items = append(result.Items, build)
		}
	}
	return result, err
}

// Get obtains the build specified by its id.
func (storage *BuildRegistryStorage) Get(id string) (interface{}, error) {
	build, err := storage.registry.GetBuild(id)
	if err != nil {
		return nil, err
	}
	return build, err
}

// Delete asynchronously deletes the Build specified by its id.
func (storage *BuildRegistryStorage) Delete(id string) (<-chan interface{}, error) {
	return apiserver.MakeAsync(func() (interface{}, error) {
		return api.Status{Status: api.StatusSuccess}, storage.registry.DeleteBuild(id)
	}), nil
}

// Extract deserializes user provided data into an api.Build.
func (storage *BuildRegistryStorage) Extract(body []byte) (interface{}, error) {
	result := api.Build{}
	err := api.DecodeInto(body, &result)
	return result, err
}

// Create registers a given new Build instance to storage.registry.
func (storage *BuildRegistryStorage) Create(obj interface{}) (<-chan interface{}, error) {
	build, ok := obj.(api.Build)
	if !ok {
		return nil, fmt.Errorf("not a build: %#v", obj)
	}
	if len(build.ID) == 0 {
		build.ID = uuid.NewUUID().String()
	}
	if len(build.Status) == 0 {
		build.Status = api.BuildNew
	}

	return apiserver.MakeAsync(func() (interface{}, error) {
		err := storage.registry.CreateBuild(build)
		if err != nil {
			return nil, err
		}
		return build, nil
	}), nil
}

// Update replaces a given Build instance with an existing instance in storage.registry.
func (storage *BuildRegistryStorage) Update(obj interface{}) (<-chan interface{}, error) {
	build, ok := obj.(api.Build)
	if !ok {
		return nil, fmt.Errorf("not a build: %#v", obj)
	}
	if len(build.ID) == 0 {
		return nil, fmt.Errorf("ID should not be empty: %#v", build)
	}
	return apiserver.MakeAsync(func() (interface{}, error) {
		err := storage.registry.UpdateBuild(build)
		if err != nil {
			return nil, err
		}
		return build, nil
	}), nil
}
