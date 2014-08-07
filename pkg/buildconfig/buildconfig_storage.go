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

package buildconfig

import (
	"fmt"
	"time"

	"code.google.com/p/go-uuid/uuid"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/buildconfig/buildconfigapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
)

// BuildConfigRegistryStorage is an implementation of RESTStorage for the api server.
type BuildConfigRegistryStorage struct {
	registry BuildConfigRegistry
}

func NewBuildConfigRegistryStorage(registry BuildConfigRegistry) apiserver.RESTStorage {
	return &BuildConfigRegistryStorage{
		registry: registry,
	}
}

// List obtains a list of BuildConfigs that match selector.
func (storage *BuildConfigRegistryStorage) List(selector labels.Selector) (interface{}, error) {
	result := buildconfigapi.BuildConfigList{}
	buildConfigs, err := storage.registry.ListBuildConfigs()
	if err == nil {
		for _, buildConfig := range buildConfigs.Items {
			result.Items = append(result.Items, buildConfig)
		}
	}
	return result, err
}

// Get obtains the BuildConfig specified by its id.
func (storage *BuildConfigRegistryStorage) Get(id string) (interface{}, error) {
	buildConfig, err := storage.registry.GetBuildConfig(id)
	if err != nil {
		return nil, err
	}
	return buildConfig, err
}

// Delete asynchronously deletes the BuildConfig specified by its id.
func (storage *BuildConfigRegistryStorage) Delete(id string) (<-chan interface{}, error) {
	return apiserver.MakeAsync(func() (interface{}, error) {
		return api.Status{Status: api.StatusSuccess}, storage.registry.DeleteBuildConfig(id)
	}), nil
}

// Extract deserializes user provided data into an buildconfigapi.BuildConfig.
func (storage *BuildConfigRegistryStorage) Extract(body []byte) (interface{}, error) {
	result := buildconfigapi.BuildConfig{}
	err := api.DecodeInto(body, &result)
	return result, err
}

// Create registers a given new BuildConfig instance to storage.registry.
func (storage *BuildConfigRegistryStorage) Create(obj interface{}) (<-chan interface{}, error) {
	buildConfig, ok := obj.(buildconfigapi.BuildConfig)
	if !ok {
		return nil, fmt.Errorf("not a build: %#v", obj)
	}
	if len(buildConfig.ID) == 0 {
		buildConfig.ID = uuid.NewUUID().String()
	}

	if buildConfig.CreationTimestamp == "" {
		buildConfig.CreationTimestamp = time.Now().Format(time.UnixDate)
	}

	return apiserver.MakeAsync(func() (interface{}, error) {
		err := storage.registry.CreateBuildConfig(buildConfig)
		if err != nil {
			return nil, err
		}
		return buildConfig, nil
	}), nil
}

// Update replaces a given BuildConfig instance with an existing instance in storage.registry.
func (storage *BuildConfigRegistryStorage) Update(obj interface{}) (<-chan interface{}, error) {
	build, ok := obj.(buildconfigapi.BuildConfig)
	if !ok {
		return nil, fmt.Errorf("not a build: %#v", obj)
	}
	if len(build.ID) == 0 {
		return nil, fmt.Errorf("ID should not be empty: %#v", build)
	}
	return apiserver.MakeAsync(func() (interface{}, error) {
		err := storage.registry.UpdateBuildConfig(build)
		if err != nil {
			return nil, err
		}
		return build, nil
	}), nil
}
