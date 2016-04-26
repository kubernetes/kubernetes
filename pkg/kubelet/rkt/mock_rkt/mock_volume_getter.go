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

// Generated via `mockgen k8s.io/kubernetes/pkg/kubelet/rkt VolumeGetter > mock_rkt/mock_volume_getter.go`
// Edited to include required boilerplate
// Source: k8s.io/kubernetes/pkg/kubelet/rkt (interfaces: VolumeGetter)

package mock_rkt

import (
	gomock "github.com/golang/mock/gomock"
	container "k8s.io/kubernetes/pkg/kubelet/container"
	types "k8s.io/kubernetes/pkg/types"
)

// Mock of VolumeGetter interface
type MockVolumeGetter struct {
	ctrl     *gomock.Controller
	recorder *_MockVolumeGetterRecorder
}

// Recorder for MockVolumeGetter (not exported)
type _MockVolumeGetterRecorder struct {
	mock *MockVolumeGetter
}

func NewMockVolumeGetter(ctrl *gomock.Controller) *MockVolumeGetter {
	mock := &MockVolumeGetter{ctrl: ctrl}
	mock.recorder = &_MockVolumeGetterRecorder{mock}
	return mock
}

func (_m *MockVolumeGetter) EXPECT() *_MockVolumeGetterRecorder {
	return _m.recorder
}

func (_m *MockVolumeGetter) GetVolumes(_param0 types.UID) (container.VolumeMap, bool) {
	ret := _m.ctrl.Call(_m, "GetVolumes", _param0)
	ret0, _ := ret[0].(container.VolumeMap)
	ret1, _ := ret[1].(bool)
	return ret0, ret1
}

func (_mr *_MockVolumeGetterRecorder) GetVolumes(arg0 interface{}) *gomock.Call {
	return _mr.mock.ctrl.RecordCall(_mr.mock, "GetVolumes", arg0)
}
