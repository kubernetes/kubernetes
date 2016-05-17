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
	"k8s.io/kubernetes/pkg/api"
)

// Mock of ImageManager interface
type MockImageManager struct {
	mock.Mock
}

func (m *MockImageManager) EnsureImageExists(pod *api.Pod, container *api.Container, pullSecrets []api.Secret) (error, string) {
	args := m.Called(pod, container, pullSecrets)
	return args.Error(0), args.String(1)
}

func (m *MockImageManager) DecImageUsage(container *api.Container) error {
	args := m.Called(container)
	return args.Error(0)
}

func (m *MockImageManager) DeleteUnusedImages() {}
