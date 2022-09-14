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

package mockfileclient

import (
	context "context"
	reflect "reflect"

	storage "github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2019-06-01/storage"
	gomock "github.com/golang/mock/gomock"
	fileclient "k8s.io/legacy-cloud-providers/azure/clients/fileclient"
)

// MockInterface is a mock of Interface interface.
type MockInterface struct {
	ctrl     *gomock.Controller
	recorder *MockInterfaceMockRecorder
}

// MockInterfaceMockRecorder is the mock recorder for MockInterface.
type MockInterfaceMockRecorder struct {
	mock *MockInterface
}

// NewMockInterface creates a new mock instance.
func NewMockInterface(ctrl *gomock.Controller) *MockInterface {
	mock := &MockInterface{ctrl: ctrl}
	mock.recorder = &MockInterfaceMockRecorder{mock}
	return mock
}

// EXPECT returns an object that allows the caller to indicate expected use.
func (m *MockInterface) EXPECT() *MockInterfaceMockRecorder {
	return m.recorder
}

// CreateFileShare mocks base method.
func (m *MockInterface) CreateFileShare(ctx context.Context, resourceGroupName, accountName string, shareOptions *fileclient.ShareOptions) error {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "CreateFileShare", ctx, resourceGroupName, accountName, shareOptions)
	ret0, _ := ret[0].(error)
	return ret0
}

// CreateFileShare indicates an expected call of CreateFileShare.
func (mr *MockInterfaceMockRecorder) CreateFileShare(ctx, resourceGroupName, accountName, shareOptions interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "CreateFileShare", reflect.TypeOf((*MockInterface)(nil).CreateFileShare), ctx, resourceGroupName, accountName, shareOptions)
}

// DeleteFileShare mocks base method.
func (m *MockInterface) DeleteFileShare(ctx context.Context, resourceGroupName, accountName, name string) error {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "DeleteFileShare", ctx, resourceGroupName, accountName, name)
	ret0, _ := ret[0].(error)
	return ret0
}

// DeleteFileShare indicates an expected call of DeleteFileShare.
func (mr *MockInterfaceMockRecorder) DeleteFileShare(ctx, resourceGroupName, accountName, name interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "DeleteFileShare", reflect.TypeOf((*MockInterface)(nil).DeleteFileShare), ctx, resourceGroupName, accountName, name)
}

// GetFileShare mocks base method.
func (m *MockInterface) GetFileShare(ctx context.Context, resourceGroupName, accountName, name string) (storage.FileShare, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetFileShare", ctx, resourceGroupName, accountName, name)
	ret0, _ := ret[0].(storage.FileShare)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// GetFileShare indicates an expected call of GetFileShare.
func (mr *MockInterfaceMockRecorder) GetFileShare(ctx, resourceGroupName, accountName, name interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetFileShare", reflect.TypeOf((*MockInterface)(nil).GetFileShare), ctx, resourceGroupName, accountName, name)
}

// ResizeFileShare mocks base method.
func (m *MockInterface) ResizeFileShare(ctx context.Context, resourceGroupName, accountName, name string, sizeGiB int) error {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "ResizeFileShare", ctx, resourceGroupName, accountName, name, sizeGiB)
	ret0, _ := ret[0].(error)
	return ret0
}

// ResizeFileShare indicates an expected call of ResizeFileShare.
func (mr *MockInterfaceMockRecorder) ResizeFileShare(ctx, resourceGroupName, accountName, name, sizeGiB interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "ResizeFileShare", reflect.TypeOf((*MockInterface)(nil).ResizeFileShare), ctx, resourceGroupName, accountName, name, sizeGiB)
}
