/*
Copyright 2024 The Kubernetes Authors.

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

package testing

import (
	"k8s.io/apimachinery/pkg/util/version"
	nodedeclaredfeatures "k8s.io/component-helpers/nodedeclaredfeatures"
)

// MockFeature is a mock implementation of nodedeclaredfeatures.Feature for testing.
type MockFeature struct {
	NameFunc            func() string
	DiscoverFunc        func(cfg *nodedeclaredfeatures.NodeConfiguration) (bool, error)
	InferFromCreateFunc func(podInfo *nodedeclaredfeatures.PodInfo) bool
	InferFromUpdateFunc func(oldPodInfo, newPodInfo *nodedeclaredfeatures.PodInfo) bool
	MinVersionFunc      func() *version.Version
	MaxVersionFunc      func() *version.Version
}

func (m *MockFeature) Name() string {
	if m.NameFunc != nil {
		return m.NameFunc()
	}
	return "mock-feature"
}

func (m *MockFeature) Discover(cfg *nodedeclaredfeatures.NodeConfiguration) (bool, error) {
	if m.DiscoverFunc != nil {
		return m.DiscoverFunc(cfg)
	}
	return false, nil
}

func (m *MockFeature) InferFromCreate(podInfo *nodedeclaredfeatures.PodInfo) bool {
	if m.InferFromCreateFunc != nil {
		return m.InferFromCreateFunc(podInfo)
	}
	return false
}

func (m *MockFeature) InferFromUpdate(oldPodInfo, newPodInfo *nodedeclaredfeatures.PodInfo) bool {
	if m.InferFromUpdateFunc != nil {
		return m.InferFromUpdateFunc(oldPodInfo, newPodInfo)
	}
	return false
}

func (m *MockFeature) MinVersion() *version.Version {
	if m.MinVersionFunc != nil {
		return m.MinVersionFunc()
	}
	return version.MustParseSemantic("0.0.0")
}

func (m *MockFeature) MaxVersion() *version.Version {
	if m.MaxVersionFunc != nil {
		return m.MaxVersionFunc()
	}
	return nil
}
