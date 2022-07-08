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

package winkernel

import (
	"github.com/stretchr/testify/mock"
	"k8s.io/kubernetes/pkg/proxy/winkernel/mocks"
)

// Mock struct created for HostNetworkService
type HnsMock struct {
	mock.Mock
}

// getNetworkByName refers to the function used for mocking hns.getNetworkByName function
// in unit testing
func (m HnsMock) getNetworkByName(name string) (*hnsNetworkInfo, error) {
	args := m.Called(name)
	return args.Get(0).(*hnsNetworkInfo), args.Error(1)
}

// getAllEndpointsByNetwork refers to the function used for mocking hns.getAllEndpointsByNetwork function
// in unit testing
func (m HnsMock) getAllEndpointsByNetwork(networkName string) (map[string]*endpointsInfo, error) {
	args := m.Called(networkName)
	return args.Get(0).(map[string]*endpointsInfo), args.Error(1)
}

// getEndpointByID refers to the function used for mocking hns.getEndpointByID function
// in unit testing
func (m HnsMock) getEndpointByID(id string) (*endpointsInfo, error) {
	args := m.Called(id)
	return args.Get(0).(*endpointsInfo), args.Error(1)
}

// getEndpointByIpAddress refers to the function used for mocking hns.getEndpointByIpAddress function
// in unit testing
func (m HnsMock) getEndpointByIpAddress(ip string, networkName string) (*endpointsInfo, error) {
	args := m.Called(ip, networkName)
	return args.Get(0).(*endpointsInfo), args.Error(1)
}

// getEndpointByName refers to the function used for mocking hns.getEndpointByName function
// in unit testing
func (m HnsMock) getEndpointByName(id string) (*endpointsInfo, error) {
	args := m.Called(id)
	return args.Get(0).(*endpointsInfo), args.Error(1)
}

// createEndpoint refers to the function used for mocking hns.createEndpoint function
// in unit testing
func (m HnsMock) createEndpoint(ep *endpointsInfo, networkName string) (*endpointsInfo, error) {
	args := m.Called(ep, networkName)
	return args.Get(0).(*endpointsInfo), args.Error(1)
}

// deleteEndpoint refers to the function used for mocking hns.deleteEndpoint function
// in unit testing
func (m HnsMock) deleteEndpoint(hnsID string) error {
	args := m.Called(hnsID)
	return args.Error(0)
}

// getLoadBalancer refers to the function used for mocking hns.getLoadBalancer function
// in unit testing
func (m HnsMock) getLoadBalancer(endpoints []endpointsInfo, flags loadBalancerFlags, sourceVip string, vip string, protocol uint16, internalPort uint16, externalPort uint16, previousLoadBalancers map[loadBalancerIdentifier]*loadBalancerInfo) (*loadBalancerInfo, error) {
	args := m.Called(endpoints, flags, sourceVip, vip, protocol, internalPort, externalPort, previousLoadBalancers)
	return args.Get(0).(*loadBalancerInfo), args.Error(1)
}

// getAllLoadBalancers refers to the function used for mocking hns.getAllLoadBalancers function
// in unit testing
func (m HnsMock) getAllLoadBalancers() (map[loadBalancerIdentifier]*loadBalancerInfo, error) {
	args := m.Called()
	return args.Get(0).(map[loadBalancerIdentifier]*loadBalancerInfo), args.Error(1)
}

// deleteLoadBalancer refers to the function used for mocking hns.deleteLoadBalancer function
// in unit testing
func (m HnsMock) deleteLoadBalancer(hnsID string) error {
	args := m.Called(hnsID)
	return args.Error(0)
}

// mockNewHNSNetworkInfo mocks hnsNetworkInfo object
func mockNewHNSNetworkInfo(id, name string) *hnsNetworkInfo {
	return &hnsNetworkInfo{
		id:            id,
		name:          name,
		networkType:   mocks.NwType,
		remoteSubnets: make([]*remoteSubnetInfo, 0),
	}
}

func mockNewEndpointInfo(hns HostNetworkService, epIp, epMac, hnsID string, flags bool) (info *endpointsInfo) {
	info = &endpointsInfo{
		ip:              epIp,
		macAddress:      epMac,
		hnsID:           hnsID,
		providerAddress: mocks.ProviderAddress,
		hns:             hns,
		isLocal:         flags,
		ready:           flags,
		serving:         flags,
		terminating:     flags,
	}
	return info
}

func mockNewHNSNetworkInfoMap(hns HostNetworkService) (eps map[string]*endpointsInfo) {
	eps = make(map[string]*endpointsInfo)
	epInfo := mockNewEndpointInfo(hns, epIpAddress, epMacAddress, mocks.HnsID, true)
	eps[epInfo.ip] = epInfo
	eps[epInfo.hnsID] = epInfo
	return eps
}

func mockNewHNSNetworkInfoList(hns HostNetworkService, ipList []string, macList []string, providerAddress string, refCount uint16) (eps []endpointsInfo) {
	for i := range ipList {
		var refCountPtr *uint16
		if i == 0 {
			refCountPtr = &refCount
		}
		epInfo := endpointsInfo{
			ip:              ipList[i],
			macAddress:      macList[i],
			providerAddress: providerAddress,
			hns:             hns,
			isLocal:         false,
			ready:           false,
			serving:         false,
			terminating:     false,
			refCount:        refCountPtr,
		}
		eps = append(eps, epInfo)
	}
	return eps
}

func mockNewLoadBalancerIdentifier() loadBalancerIdentifier {
	return loadBalancerIdentifier{protocol: uint16(protocol), internalPort: uint16(internalPort), externalPort: uint16(externalPort)}
}

func mockNewAllLoadBalancers() (lbs map[loadBalancerIdentifier]*loadBalancerInfo) {
	lbs = make(map[loadBalancerIdentifier]*loadBalancerInfo)
	id := mockNewLoadBalancerIdentifier()
	lbs[id] = &loadBalancerInfo{
		hnsID: mocks.HnsID,
	}
	return lbs
}

func mockCreateEndpoint(mockHns *HnsMock, epIp, epMac, providerIpAddress, hnsID string, flags bool) *HnsMock {
	mockEndpointInfo := mockNewEndpointInfo(nil, epIp, epMac, hnsID, flags)
	mockEndpointInfo.providerAddress = providerIpAddress
	mockHns.On("createEndpoint", mockEndpointInfo, mocks.TestNwName).Return(mockEndpointInfo, nil)
	return mockHns
}
