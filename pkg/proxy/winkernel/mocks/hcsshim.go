//go:build windows
// +build windows

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

package mocks

import (
	"encoding/json"

	"github.com/Microsoft/hcsshim"
	"github.com/stretchr/testify/mock"
)

// Mock struct created for hcsshim package
type HcsshimMock struct {
	mock.Mock
}

// HNSListPolicyListRequest refers to the function used for mocking hcsshim.HNSListPolicyListRequest
// in unit testing
func (m *HcsshimMock) HNSListPolicyListRequest() ([]hcsshim.PolicyList, error) {
	args := m.Called()
	return args.Get(0).([]hcsshim.PolicyList), args.Error(1)
}

// GetHNSNetworkByName refers to the function used for mocking hcsshim.GetHNSNetworkByName
// in unit testing
func (m *HcsshimMock) GetHNSNetworkByName(networkName string) (*hcsshim.HNSNetwork, error) {
	args := m.Called(networkName)
	return args.Get(0).(*hcsshim.HNSNetwork), args.Error(1)
}

// MockNewHNSNetwork create a mock object for HNSNetwork
func MockNewHNSNetwork(id, name string) (obj *hcsshim.HNSNetwork) {
	if id == "" || id == "nil" {
		obj = nil
		return obj
	}
	obj = new(hcsshim.HNSNetwork)
	obj.Id = id
	obj.Name = name
	obj.NetworkAdapterName = NwAdapterName
	obj.Policies = make([]json.RawMessage, 0)
	obj.MacPools = make([]hcsshim.MacPool, 0)
	obj.Subnets = make([]hcsshim.Subnet, 0)
	return obj
}
