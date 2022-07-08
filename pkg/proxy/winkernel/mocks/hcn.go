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
	"github.com/Microsoft/hcsshim/hcn"
	"github.com/stretchr/testify/mock"
)

// Mock struct created for hcn package
type HcnMock struct {
	mock.Mock
}

// GetSupportedFeatures refers to the function used for mocking hcn.GetSupportedFeatures
// in unit testing
func (m *HcnMock) GetSupportedFeatures() hcn.SupportedFeatures {
	args := m.Called()
	return args.Get(0).(hcn.SupportedFeatures)
}

// IPv6DualStackSupported refers to the function used for mocking hcn.IPv6DualStackSupported
// in unit testing
func (m *HcnMock) IPv6DualStackSupported() error {
	args := m.Called()
	return args.Error(0)
}

// DSRSupported refers to the function used for mocking hcn.DSRSupported
// in unit testing
func (m *HcnMock) DSRSupported() error {
	args := m.Called()
	return args.Error(0)
}

// RemoteSubnetSupported refers to the function used for mocking hcn.RemoteSubnetSupported
// in unit testing
func (m *HcnMock) RemoteSubnetSupported() error {
	args := m.Called()
	return args.Error(0)
}

// MockNewSupportedFeatures create a mock object for SupportedFeatures
func MockNewSupportedFeatures() hcn.SupportedFeatures {
	return hcn.SupportedFeatures{
		Acl:          hcn.AclFeatures{},
		Api:          hcn.ApiSupport{},
		RemoteSubnet: true,
		HostRoute:    true,
		DSR:          true,
	}
}
