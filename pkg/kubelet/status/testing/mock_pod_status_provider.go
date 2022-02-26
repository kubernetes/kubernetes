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
	reflect "reflect"

	gomock "github.com/golang/mock/gomock"
	"github.com/stretchr/testify/mock"
	v1 "k8s.io/api/core/v1"
	types "k8s.io/apimachinery/pkg/types"
)

// MockPodStatusProvider is a mock of PodStatusProvider interface
type MockPodStatusProvider struct {
	ctrl     *gomock.Controller
	recorder *MockPodStatusProviderMockRecorder
}

// MockPodStatusProviderMockRecorder is the mock recorder for MockPodStatusProvider
type MockPodStatusProviderMockRecorder struct {
	mock *MockPodStatusProvider
}

// NewMockPodStatusProvider creates a new mock instance
func NewMockPodStatusProvider(ctrl *gomock.Controller) *MockPodStatusProvider {
	mock := &MockPodStatusProvider{ctrl: ctrl}
	mock.recorder = &MockPodStatusProviderMockRecorder{mock}
	return mock
}

// EXPECT returns an object that allows the caller to indicate expected use
func (m *MockPodStatusProvider) EXPECT() *MockPodStatusProviderMockRecorder {
	return m.recorder
}

// GetPodStatus mocks base method
func (m *MockPodStatusProvider) GetPodStatus(uid types.UID) (v1.PodStatus, bool) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetPodStatus", uid)
	ret0, _ := ret[0].(v1.PodStatus)
	ret1, _ := ret[1].(bool)
	return ret0, ret1
}

// GetPodStatus indicates an expected call of GetPodStatus
func (mr *MockPodStatusProviderMockRecorder) GetPodStatus(uid interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetPodStatus", reflect.TypeOf((*MockPodStatusProvider)(nil).GetPodStatus), uid)
}

// MockPodDeletionSafetyProvider is a mock of PodDeletionSafetyProvider interface
type MockPodDeletionSafetyProvider struct {
	ctrl     *gomock.Controller
	recorder *MockPodDeletionSafetyProviderMockRecorder
}

// MockPodDeletionSafetyProviderMockRecorder is the mock recorder for MockPodDeletionSafetyProvider
type MockPodDeletionSafetyProviderMockRecorder struct {
	mock *MockPodDeletionSafetyProvider
}

// NewMockPodDeletionSafetyProvider creates a new mock instance
func NewMockPodDeletionSafetyProvider(ctrl *gomock.Controller) *MockPodDeletionSafetyProvider {
	mock := &MockPodDeletionSafetyProvider{ctrl: ctrl}
	mock.recorder = &MockPodDeletionSafetyProviderMockRecorder{mock}
	return mock
}

// EXPECT returns an object that allows the caller to indicate expected use
func (m *MockPodDeletionSafetyProvider) EXPECT() *MockPodDeletionSafetyProviderMockRecorder {
	return m.recorder
}

// PodResourcesAreReclaimed mocks base method
func (m *MockPodDeletionSafetyProvider) PodResourcesAreReclaimed(pod *v1.Pod, status v1.PodStatus) bool {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "PodResourcesAreReclaimed", pod, status)
	ret0, _ := ret[0].(bool)
	return ret0
}

// PodResourcesAreReclaimed indicates an expected call of PodResourcesAreReclaimed
func (mr *MockPodDeletionSafetyProviderMockRecorder) PodResourcesAreReclaimed(pod, status interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "PodResourcesAreReclaimed", reflect.TypeOf((*MockPodDeletionSafetyProvider)(nil).PodResourcesAreReclaimed), pod, status)
}

// PodCouldHaveRunningContainers mocks base method
func (m *MockPodDeletionSafetyProvider) PodCouldHaveRunningContainers(pod *v1.Pod) bool {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "PodCouldHaveRunningContainers", pod)
	ret0, _ := ret[0].(bool)
	return ret0
}

// PodCouldHaveRunningContainers indicates an expected call of PodCouldHaveRunningContainers
func (mr *MockPodDeletionSafetyProviderMockRecorder) PodCouldHaveRunningContainers(pod interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "PodCouldHaveRunningContainers", reflect.TypeOf((*MockPodDeletionSafetyProvider)(nil).PodCouldHaveRunningContainers), pod)
}

// MockManager is a mock of Manager interface
type MockManager struct {
	ctrl     *gomock.Controller
	recorder *MockManagerMockRecorder
}

// MockManagerMockRecorder is the mock recorder for MockManager
type MockManagerMockRecorder struct {
	mock *MockManager
}

// NewMockManager creates a new mock instance
func NewMockManager(ctrl *gomock.Controller) *MockManager {
	mock := &MockManager{ctrl: ctrl}
	mock.recorder = &MockManagerMockRecorder{mock}
	return mock
}

// EXPECT returns an object that allows the caller to indicate expected use
func (m *MockManager) EXPECT() *MockManagerMockRecorder {
	return m.recorder
}

// GetPodStatus mocks base method
func (m *MockManager) GetPodStatus(uid types.UID) (v1.PodStatus, bool) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetPodStatus", uid)
	ret0, _ := ret[0].(v1.PodStatus)
	ret1, _ := ret[1].(bool)
	return ret0, ret1
}

// GetPodStatus indicates an expected call of GetPodStatus
func (mr *MockManagerMockRecorder) GetPodStatus(uid interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetPodStatus", reflect.TypeOf((*MockManager)(nil).GetPodStatus), uid)
}

// Start mocks base method
func (m *MockManager) Start() {
	m.ctrl.T.Helper()
	m.ctrl.Call(m, "Start")
}

// Start indicates an expected call of Start
func (mr *MockManagerMockRecorder) Start() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Start", reflect.TypeOf((*MockManager)(nil).Start))
}

// SetPodStatus mocks base method
func (m *MockManager) SetPodStatus(pod *v1.Pod, status v1.PodStatus) {
	m.ctrl.T.Helper()
	m.ctrl.Call(m, "SetPodStatus", pod, status)
}

// MockStatusProvider mocks a PodStatusProvider.
type MockStatusProvider struct {
	mock.Mock
}

// GetPodStatus implements PodStatusProvider.
func (m *MockStatusProvider) GetPodStatus(uid types.UID) (v1.PodStatus, bool) {
	args := m.Called(uid)
	return args.Get(0).(v1.PodStatus), args.Bool(1)
}
