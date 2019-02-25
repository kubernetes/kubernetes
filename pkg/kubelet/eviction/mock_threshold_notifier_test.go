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

package eviction

import (
	mock "github.com/stretchr/testify/mock"
	statsapi "k8s.io/kubernetes/pkg/kubelet/apis/stats/v1alpha1"
)

// MockCgroupNotifier is a mock implementation of the CgroupNotifier interface
type MockCgroupNotifier struct {
	mock.Mock
}

// Start implements the NotifierFactory interface
func (m *MockCgroupNotifier) Start(a0 chan<- struct{}) {
	m.Called(a0)
}

// Stop implements the NotifierFactory interface
func (m *MockCgroupNotifier) Stop() {
	m.Called()
}

// MockNotifierFactory is a mock of the NotifierFactory interface
type MockNotifierFactory struct {
	mock.Mock
}

// NewCgroupNotifier implements the NotifierFactory interface
func (m *MockNotifierFactory) NewCgroupNotifier(a0, a1 string, a2 int64) (CgroupNotifier, error) {
	ret := m.Called(a0, a1, a2)

	var r0 CgroupNotifier
	if rf, ok := ret.Get(0).(func(string, string, int64) CgroupNotifier); ok {
		r0 = rf(a0, a1, a2)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(CgroupNotifier)
		}
	}
	var r1 error
	if rf, ok := ret.Get(1).(func(string, string, int64) error); ok {
		r1 = rf(a0, a1, a2)
	} else {
		r1 = ret.Error(1)
	}
	return r0, r1
}

// MockThresholdNotifier is a mock implementation of the ThresholdNotifier interface
type MockThresholdNotifier struct {
	mock.Mock
}

// Start implements the ThresholdNotifier interface
func (m *MockThresholdNotifier) Start() {
	m.Called()
}

// UpdateThreshold implements the ThresholdNotifier interface
func (m *MockThresholdNotifier) UpdateThreshold(a0 *statsapi.Summary) error {
	ret := m.Called(a0)

	var r0 error
	if rf, ok := ret.Get(0).(func(*statsapi.Summary) error); ok {
		r0 = rf(a0)
	} else {
		r0 = ret.Error(0)
	}
	return r0
}

// Description implements the ThresholdNotifier interface
func (m *MockThresholdNotifier) Description() string {
	ret := m.Called()
	var r0 string
	if rf, ok := ret.Get(0).(func() string); ok {
		r0 = rf()
	} else {
		r0 = ret.String(0)
	}
	return r0
}
