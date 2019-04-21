/*
Copyright 2016 The Kubernetes Authors.

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

// Generated via: mockgen os FileInfo
// Edited to include required boilerplate
// Source: os (interfaces: FileInfo)

package testing

import (
	os "os"
	time "time"

	gomock "github.com/golang/mock/gomock"
)

// MockFileInfo is a Mock of the FileInfo interface
type MockFileInfo struct {
	ctrl     *gomock.Controller
	recorder *mockFileInfoRecorder
}

// mockFileInfoRecorder is a Recorder for MockFileInfo
type mockFileInfoRecorder struct {
	mock *MockFileInfo
}

// NewMockFileInfo creates a MockFileInfo.
func NewMockFileInfo(ctrl *gomock.Controller) *MockFileInfo {
	mock := &MockFileInfo{ctrl: ctrl}
	mock.recorder = &mockFileInfoRecorder{mock}
	return mock
}

// EXPECT returns a mocked FileInfoRecorder of the MockFileInfo object.
func (_m *MockFileInfo) EXPECT() *mockFileInfoRecorder {
	return _m.recorder
}

// IsDir checks for directory existence on MockFileInfo.
func (_m *MockFileInfo) IsDir() bool {
	ret := _m.ctrl.Call(_m, "IsDir")
	ret0, _ := ret[0].(bool)
	return ret0
}

// IsDir checks for directory existence on mockFileInfoRecorder.
func (_mr *mockFileInfoRecorder) IsDir() *gomock.Call {
	return _mr.mock.ctrl.RecordCall(_mr.mock, "IsDir")
}

// ModTime returns ModTime from MockFileInfo.
func (_m *MockFileInfo) ModTime() time.Time {
	ret := _m.ctrl.Call(_m, "ModTime")
	ret0, _ := ret[0].(time.Time)
	return ret0
}

// ModTime returns ModTime from MockFileInfoRecroder.
func (_mr *mockFileInfoRecorder) ModTime() *gomock.Call {
	return _mr.mock.ctrl.RecordCall(_mr.mock, "ModTime")
}

// Mode returns the FileMode from MockFileInfo.
func (_m *MockFileInfo) Mode() os.FileMode {
	ret := _m.ctrl.Call(_m, "Mode")
	ret0, _ := ret[0].(os.FileMode)
	return ret0
}

// Mode returns the FileMode from mockFileInfoRecorder.
func (_mr *mockFileInfoRecorder) Mode() *gomock.Call {
	return _mr.mock.ctrl.RecordCall(_mr.mock, "Mode")
}

// Name returns the Name from MockFileInfo.
func (_m *MockFileInfo) Name() string {
	ret := _m.ctrl.Call(_m, "Name")
	ret0, _ := ret[0].(string)
	return ret0
}

// Name returns the Name from mockFileInfoRecorder.
func (_mr *mockFileInfoRecorder) Name() *gomock.Call {
	return _mr.mock.ctrl.RecordCall(_mr.mock, "Name")
}

// Size returns the Size from MockFileInfo.
func (_m *MockFileInfo) Size() int64 {
	ret := _m.ctrl.Call(_m, "Size")
	ret0, _ := ret[0].(int64)
	return ret0
}

// Size returns the Size from mockFileInfoRecorder.
func (_mr *mockFileInfoRecorder) Size() *gomock.Call {
	return _mr.mock.ctrl.RecordCall(_mr.mock, "Size")
}

// Sys returns an empty interface from MockFileInfo.
func (_m *MockFileInfo) Sys() interface{} {
	ret := _m.ctrl.Call(_m, "Sys")
	ret0, _ := ret[0].(interface{})
	return ret0
}

// Sys returns an empty interface from mockFileInfoRecorder.
func (_mr *mockFileInfoRecorder) Sys() *gomock.Call {
	return _mr.mock.ctrl.RecordCall(_mr.mock, "Sys")
}
