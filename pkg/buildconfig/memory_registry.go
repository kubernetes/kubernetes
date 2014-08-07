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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/buildconfig/buildconfigapi"
)

// An implementation of BuildConfigRegistry that is backed by memory
// Mainly used for testing.
type MemoryRegistry struct {
	buildConfigData map[string]buildconfigapi.BuildConfig
}

func MakeMemoryRegistry() *MemoryRegistry {
	return &MemoryRegistry{
		buildConfigData: map[string]buildconfigapi.BuildConfig{},
	}
}

func (registry *MemoryRegistry) ListBuildConfigs() (buildconfigapi.BuildConfigList, error) {
	result := []buildconfigapi.BuildConfig{}
	for _, value := range registry.buildConfigData {
		result = append(result, value)
	}
	return buildconfigapi.BuildConfigList{Items: result}, nil
}

func (registry *MemoryRegistry) GetBuildConfig(buildConfigID string) (*buildconfigapi.BuildConfig, error) {
	build, found := registry.buildConfigData[buildConfigID]
	if found {
		return &build, nil
	} else {
		return nil, apiserver.NewNotFoundErr("buildconfig", buildConfigID)
	}
}

func (registry *MemoryRegistry) CreateBuildConfig(buildConfig buildconfigapi.BuildConfig) error {
	registry.buildConfigData[buildConfig.ID] = buildConfig
	return nil
}

func (registry *MemoryRegistry) DeleteBuildConfig(buildConfigID string) error {
	if _, ok := registry.buildConfigData[buildConfigID]; !ok {
		return apiserver.NewNotFoundErr("buildconfig", buildConfigID)
	}
	delete(registry.buildConfigData, buildConfigID)
	return nil
}

func (registry *MemoryRegistry) UpdateBuildConfig(buildConfig buildconfigapi.BuildConfig) error {
	if _, ok := registry.buildConfigData[buildConfig.ID]; !ok {
		return apiserver.NewNotFoundErr("buildconfig", buildConfig.ID)
	}
	registry.buildConfigData[buildConfig.ID] = buildConfig
	return nil
}
