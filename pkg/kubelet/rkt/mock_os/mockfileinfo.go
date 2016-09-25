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

package mock_os

import (
	os "os"
	time "time"

	gomock "github.com/golang/mock/gomock"
)

// Mock of FileInfo interface
type MockFileInfo struct {
	ctrl     *gomock.Controller
	recorder *_MockFileInfoRecorder
}

// Recorder for MockFileInfo (not exported)
type _MockFileInfoRecorder struct {
	mock *MockFileInfo
}

func NewMockFileInfo(ctrl *gomock.Controller) *MockFileInfo {
	mock := &MockFileInfo{ctrl: ctrl}
	mock.recorder = &_MockFileInfoRecorder{mock}
	return mock
}

func (_m *MockFileInfo) EXPECT() *_MockFileInfoRecorder {
	return _m.recorder
}

func (_m *MockFileInfo) IsDir() bool {
	ret := _m.ctrl.Call(_m, "IsDir")
	ret0, _ := ret[0].(bool)
	return ret0
}

func (_mr *_MockFileInfoRecorder) IsDir() *gomock.Call {
	return _mr.mock.ctrl.RecordCall(_mr.mock, "IsDir")
}

func (_m *MockFileInfo) ModTime() time.Time {
	ret := _m.ctrl.Call(_m, "ModTime")
	ret0, _ := ret[0].(time.Time)
	return ret0
}

func (_mr *_MockFileInfoRecorder) ModTime() *gomock.Call {
	return _mr.mock.ctrl.RecordCall(_mr.mock, "ModTime")
}

func (_m *MockFileInfo) Mode() os.FileMode {
	ret := _m.ctrl.Call(_m, "Mode")
	ret0, _ := ret[0].(os.FileMode)
	return ret0
}

func (_mr *_MockFileInfoRecorder) Mode() *gomock.Call {
	return _mr.mock.ctrl.RecordCall(_mr.mock, "Mode")
}

func (_m *MockFileInfo) Name() string {
	ret := _m.ctrl.Call(_m, "Name")
	ret0, _ := ret[0].(string)
	return ret0
}

func (_mr *_MockFileInfoRecorder) Name() *gomock.Call {
	return _mr.mock.ctrl.RecordCall(_mr.mock, "Name")
}

func (_m *MockFileInfo) Size() int64 {
	ret := _m.ctrl.Call(_m, "Size")
	ret0, _ := ret[0].(int64)
	return ret0
}

func (_mr *_MockFileInfoRecorder) Size() *gomock.Call {
	return _mr.mock.ctrl.RecordCall(_mr.mock, "Size")
}

func (_m *MockFileInfo) Sys() interface{} {
	ret := _m.ctrl.Call(_m, "Sys")
	ret0, _ := ret[0].(interface{})
	return ret0
}

func (_mr *_MockFileInfoRecorder) Sys() *gomock.Call {
	return _mr.mock.ctrl.RecordCall(_mr.mock, "Sys")
}
