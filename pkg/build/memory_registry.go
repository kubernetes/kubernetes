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

package build

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/build/buildapi"
)

// An implementation of BuildRegistry that is backed by memory
// Mainly used for testing.
type MemoryRegistry struct {
	buildData map[string]buildapi.Build
}

func MakeMemoryRegistry() *MemoryRegistry {
	return &MemoryRegistry{
		buildData: map[string]buildapi.Build{},
	}
}

func (registry *MemoryRegistry) ListBuilds() (buildapi.BuildList, error) {
	result := []buildapi.Build{}
	for _, value := range registry.buildData {
		result = append(result, value)
	}
	return buildapi.BuildList{Items: result}, nil
}

func (registry *MemoryRegistry) GetBuild(buildID string) (*buildapi.Build, error) {
	build, found := registry.buildData[buildID]
	if found {
		return &build, nil
	} else {
		return nil, apiserver.NewNotFoundErr("build", buildID)
	}
}

func (registry *MemoryRegistry) CreateBuild(build buildapi.Build) error {
	registry.buildData[build.ID] = build
	return nil
}

func (registry *MemoryRegistry) DeleteBuild(buildID string) error {
	if _, ok := registry.buildData[buildID]; !ok {
		return apiserver.NewNotFoundErr("build", buildID)
	}
	delete(registry.buildData, buildID)
	return nil
}

func (registry *MemoryRegistry) UpdateBuild(build buildapi.Build) error {
	if _, ok := registry.buildData[build.ID]; !ok {
		return apiserver.NewNotFoundErr("build", build.ID)
	}
	registry.buildData[build.ID] = build
	return nil
}
