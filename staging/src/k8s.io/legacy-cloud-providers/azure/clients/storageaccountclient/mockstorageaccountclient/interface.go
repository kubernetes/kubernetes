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

package mockstorageaccountclient

import (
	context "context"
	storage "github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2019-06-01/storage"
	gomock "github.com/golang/mock/gomock"
	retry "k8s.io/legacy-cloud-providers/azure/retry"
	reflect "reflect"
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

// Create mocks base method
func (m *MockInterface) Create(ctx context.Context, resourceGroupName, accountName string, parameters storage.AccountCreateParameters) *retry.Error {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "Create", ctx, resourceGroupName, accountName, parameters)
	ret0, _ := ret[0].(*retry.Error)
	return ret0
}

// Create indicates an expected call of Create
func (mr *MockInterfaceMockRecorder) Create(ctx, resourceGroupName, accountName, parameters interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Create", reflect.TypeOf((*MockInterface)(nil).Create), ctx, resourceGroupName, accountName, parameters)
}

// Delete mocks base method
func (m *MockInterface) Delete(ctx context.Context, resourceGroupName, accountName string) *retry.Error {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "Delete", ctx, resourceGroupName, accountName)
	ret0, _ := ret[0].(*retry.Error)
	return ret0
}

// Delete indicates an expected call of Delete
func (mr *MockInterfaceMockRecorder) Delete(ctx, resourceGroupName, accountName interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Delete", reflect.TypeOf((*MockInterface)(nil).Delete), ctx, resourceGroupName, accountName)
}

// ListKeys mocks base method
func (m *MockInterface) ListKeys(ctx context.Context, resourceGroupName, accountName string) (storage.AccountListKeysResult, *retry.Error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "ListKeys", ctx, resourceGroupName, accountName)
	ret0, _ := ret[0].(storage.AccountListKeysResult)
	ret1, _ := ret[1].(*retry.Error)
	return ret0, ret1
}

// ListKeys indicates an expected call of ListKeys
func (mr *MockInterfaceMockRecorder) ListKeys(ctx, resourceGroupName, accountName interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "ListKeys", reflect.TypeOf((*MockInterface)(nil).ListKeys), ctx, resourceGroupName, accountName)
}

// ListByResourceGroup mocks base method
func (m *MockInterface) ListByResourceGroup(ctx context.Context, resourceGroupName string) ([]storage.Account, *retry.Error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "ListByResourceGroup", ctx, resourceGroupName)
	ret0, _ := ret[0].([]storage.Account)
	ret1, _ := ret[1].(*retry.Error)
	return ret0, ret1
}

// ListByResourceGroup indicates an expected call of ListByResourceGroup
func (mr *MockInterfaceMockRecorder) ListByResourceGroup(ctx, resourceGroupName interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "ListByResourceGroup", reflect.TypeOf((*MockInterface)(nil).ListByResourceGroup), ctx, resourceGroupName)
}

// GetProperties mocks base method
func (m *MockInterface) GetProperties(ctx context.Context, resourceGroupName, accountName string) (storage.Account, *retry.Error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetProperties", ctx, resourceGroupName, accountName)
	ret0, _ := ret[0].(storage.Account)
	ret1, _ := ret[1].(*retry.Error)
	return ret0, ret1
}

// GetProperties indicates an expected call of GetProperties
func (mr *MockInterfaceMockRecorder) GetProperties(ctx, resourceGroupName, accountName interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetProperties", reflect.TypeOf((*MockInterface)(nil).GetProperties), ctx, resourceGroupName, accountName)
}
