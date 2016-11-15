/*
Copyright 2014 The Kubernetes Authors.

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

// mock_cni is a mock of the `libcni.CNI` interface. It's a handwritten mock
// because there are only two functions to deal with.
package mock_cni

import (
	"github.com/containernetworking/cni/libcni"
	"github.com/containernetworking/cni/pkg/types"
	"github.com/stretchr/testify/mock"
)

type MockCNI struct {
	mock.Mock
}

func (m *MockCNI) AddNetwork(net *libcni.NetworkConfig, rt *libcni.RuntimeConf) (*types.Result, error) {
	args := m.Called(net, rt)
	return args.Get(0).(*types.Result), args.Error(1)
}

func (m *MockCNI) DelNetwork(net *libcni.NetworkConfig, rt *libcni.RuntimeConf) error {
	args := m.Called(net, rt)
	return args.Error(0)
}
