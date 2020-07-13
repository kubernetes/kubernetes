// +build !providerless

/*
Copyright 2020 The Kubernetes Authors.

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

package mockdiskclient

import (
	context "context"
	reflect "reflect"

	compute "github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-12-01/compute"
	gomock "github.com/golang/mock/gomock"
	retry "k8s.io/legacy-cloud-providers/azure/retry"
)

// MockInterface is a mock of Interface interface
type MockInterface struct {
	ctrl     *gomock.Controller
	recorder *MockInterfaceMockRecorder
}

// MockInterfaceMockRecorder is the mock recorder for MockInterface
type MockInterfaceMockRecorder struct {
	mock *MockInterface
}

// NewMockInterface creates a new mock instance
func NewMockInterface(ctrl *gomock.Controller) *MockInterface {
	mock := &MockInterface{ctrl: ctrl}
	mock.recorder = &MockInterfaceMockRecorder{mock}
	return mock
}

// EXPECT returns an object that allows the caller to indicate expected use
func (m *MockInterface) EXPECT() *MockInterfaceMockRecorder {
	return m.recorder
}

// Get mocks base method
func (m *MockInterface) Get(ctx context.Context, resourceGroupName, diskName string) (compute.Disk, *retry.Error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "Get", ctx, resourceGroupName, diskName)
	ret0, _ := ret[0].(compute.Disk)
	ret1, _ := ret[1].(*retry.Error)
	return ret0, ret1
}

// Get indicates an expected call of Get
func (mr *MockInterfaceMockRecorder) Get(ctx, resourceGroupName, diskName interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Get", reflect.TypeOf((*MockInterface)(nil).Get), ctx, resourceGroupName, diskName)
}

// CreateOrUpdate mocks base method
func (m *MockInterface) CreateOrUpdate(ctx context.Context, resourceGroupName, diskName string, diskParameter compute.Disk) *retry.Error {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "CreateOrUpdate", ctx, resourceGroupName, diskName, diskParameter)
	ret0, _ := ret[0].(*retry.Error)
	return ret0
}

// CreateOrUpdate indicates an expected call of CreateOrUpdate
func (mr *MockInterfaceMockRecorder) CreateOrUpdate(ctx, resourceGroupName, diskName, diskParameter interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "CreateOrUpdate", reflect.TypeOf((*MockInterface)(nil).CreateOrUpdate), ctx, resourceGroupName, diskName, diskParameter)
}

// Update mocks base method
func (m *MockInterface) Update(ctx context.Context, resourceGroupName, diskName string, diskParameter compute.DiskUpdate) *retry.Error {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "Update", ctx, resourceGroupName, diskName, diskParameter)
	ret0, _ := ret[0].(*retry.Error)
	return ret0
}

// Update indicates an expected call of Update
func (mr *MockInterfaceMockRecorder) Update(ctx, resourceGroupName, diskName, diskParameter interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Update", reflect.TypeOf((*MockInterface)(nil).Update), ctx, resourceGroupName, diskName, diskParameter)
}

// Delete mocks base method
func (m *MockInterface) Delete(ctx context.Context, resourceGroupName, diskName string) *retry.Error {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "Delete", ctx, resourceGroupName, diskName)
	ret0, _ := ret[0].(*retry.Error)
	return ret0
}

// Delete indicates an expected call of Delete
func (mr *MockInterfaceMockRecorder) Delete(ctx, resourceGroupName, diskName interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Delete", reflect.TypeOf((*MockInterface)(nil).Delete), ctx, resourceGroupName, diskName)
}

// ListByResourceGroup mocks base method
func (m *MockInterface) ListByResourceGroup(ctx context.Context, resourceGroupName string) ([]compute.Disk, *retry.Error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "ListByResourceGroup", ctx, resourceGroupName)
	ret0, _ := ret[0].([]compute.Disk)
	ret1, _ := ret[1].(*retry.Error)
	return ret0, ret1
}

// ListByResourceGroup indicates an expected call of ListByResourceGroup
func (mr *MockInterfaceMockRecorder) ListByResourceGroup(ctx, resourceGroupName interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "ListByResourceGroup", reflect.TypeOf((*MockInterface)(nil).Delete), ctx, resourceGroupName)
}
