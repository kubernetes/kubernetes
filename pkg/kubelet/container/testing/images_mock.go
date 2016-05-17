/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"github.com/stretchr/testify/mock"
	api "k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/kubelet/container"
)

// Mock of RuntimeImages interface
type MockRuntimeImages struct {
	mock.Mock
}

var _ container.RuntimeImages = new(MockRuntimeImages)

func (m *MockRuntimeImages) PullImage(image container.ImageSpec, secrets []api.Secret) error {
	args := m.Called(image, secrets)
	return args.Error(0)
}

func (m *MockRuntimeImages) IsImagePresent(image container.ImageSpec) (bool, error) {
	args := m.Called(image)
	return args.Bool(0), args.Error(1)
}

func (m *MockRuntimeImages) ListImages() ([]container.Image, error) {
	args := m.Called()
	return args.Get(0).([]container.Image), args.Error(1)
}

func (m *MockRuntimeImages) RemoveImage(image container.ImageSpec) error {
	args := m.Called(image)
	return args.Error(0)
}

func (m *MockRuntimeImages) ImageStats() (*container.ImageStats, error) {
	args := m.Called()
	return args.Get(0).(*container.ImageStats), args.Error(1)
}
