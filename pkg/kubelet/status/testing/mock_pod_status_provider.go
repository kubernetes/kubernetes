/*
Copyright 2018 The Kubernetes Authors.

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

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
)

// MockStatusProvider mocks a PodStatusProvider.
type MockStatusProvider struct {
	mock.Mock
}

// GetPodStatus implements PodStatusProvider.
func (m *MockStatusProvider) GetPodStatus(uid types.UID) (v1.PodStatus, bool) {
	args := m.Called(uid)
	return args.Get(0).(v1.PodStatus), args.Bool(1)
}
