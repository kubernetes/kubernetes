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
	"encoding/json"

	"github.com/Microsoft/hcsshim/hcn"

	"strings"
	"testing"
)

const sourceVip = "192.168.1.2"
const serviceVip = "11.0.0.1"
const addressPrefix = "192.168.1.0/24"
const gatewayAddress = "192.168.1.1"
const epMacAddress = "00-11-22-33-44-55"
const epIpAddress = "192.168.1.3"
const epIpAddressRemote = "192.168.2.3"
const epPaAddress = "10.0.0.3"
const protocol = 6
const internalPort = 80
const externalPort = 32440

func TestGetNetworkByName(t *testing.T) {
	hnsV1 := hnsV1{}
	hnsV2 := hnsV2{}

	testGetNetworkByName(t, hnsV1)
	testGetNetworkByName(t, hnsV2)
}
func TestGetEndpointByID(t *testing.T) {
	hnsV1 := hnsV1{}
	hnsV2 := hnsV2{}

	testGetEndpointByID(t, hnsV1)
	testGetEndpointByID(t, hnsV2)
}
func TestGetEndpointByIpAddress(t *testing.T) {
	hnsV1 := hnsV1{}
	hnsV2 := hnsV2{}

	testGetEndpointByIpAddress(t, hnsV1)
	testGetEndpointByIpAddress(t, hnsV2)
}
func TestCreateEndpointLocal(t *testing.T) {
	hnsV1 := hnsV1{}
	hnsV2 := hnsV2{}

	testCreateEndpointLocal(t, hnsV1)
	testCreateEndpointLocal(t, hnsV2)
}
func TestCreateEndpointRemotePA(t *testing.T) {
	hnsV1 := hnsV1{}
	hnsV2 := hnsV2{}

	testCreateEndpointRemote(t, hnsV1, epPaAddress)
	testCreateEndpointRemote(t, hnsV2, epPaAddress)
}
func TestCreateEndpointRemoteNoPA(t *testing.T) {
	hnsV1 := hnsV1{}
	hnsV2 := hnsV2{}

	testCreateEndpointRemote(t, hnsV1, "")
	testCreateEndpointRemote(t, hnsV2, "")
}
func TestDeleteEndpoint(t *testing.T) {
	hnsV1 := hnsV1{}
	hnsV2 := hnsV2{}

	testDeleteEndpoint(t, hnsV1)
	testDeleteEndpoint(t, hnsV2)
}
func TestGetLoadBalancerExisting(t *testing.T) {
	hnsV1 := hnsV1{}
	hnsV2 := hnsV2{}

	testGetLoadBalancerExisting(t, hnsV1)
	testGetLoadBalancerExisting(t, hnsV2)
}
func TestGetLoadBalancerNew(t *testing.T) {
	hnsV1 := hnsV1{}
	hnsV2 := hnsV2{}

	testGetLoadBalancerNew(t, hnsV1)
	testGetLoadBalancerNew(t, hnsV2)
}
func TestDeleteLoadBalancer(t *testing.T) {
	hnsV1 := hnsV1{}
	hnsV2 := hnsV2{}

	testDeleteLoadBalancer(t, hnsV1)
	testDeleteLoadBalancer(t, hnsV2)
}
func testGetNetworkByName(t *testing.T, hns HostNetworkService) {
	Network, err := createTestNetwork()
	if err != nil {
		t.Error(err)
	}

	network, err := hns.getNetworkByName(Network.Name)
	if err != nil {
		t.Error(err)
	}

	if !strings.EqualFold(network.id, Network.Id) {
		t.Errorf("%v does not match %v", network.id, Network.Id)
	}
	err = Network.Delete()
	if err != nil {
		t.Error(err)
	}
}
func testGetEndpointByID(t *testing.T, hns HostNetworkService) {
	Network, err := createTestNetwork()
	if err != nil {
		t.Error(err)
	}

	ipConfig := &hcn.IpConfig{
		IpAddress: epIpAddress,
	}
	Endpoint := &hcn.HostComputeEndpoint{
		IpConfigurations: []hcn.IpConfig{*ipConfig},
		MacAddress:       epMacAddress,
		SchemaVersion: hcn.SchemaVersion{
			Major: 2,
			Minor: 0,
		},
	}

	Endpoint, err = Network.CreateEndpoint(Endpoint)
	if err != nil {
		t.Error(err)
	}

	endpoint, err := hns.getEndpointByID(Endpoint.Id)
	if err != nil {
		t.Error(err)
	}
	if !strings.EqualFold(endpoint.hnsID, Endpoint.Id) {
		t.Errorf("%v does not match %v", endpoint.hnsID, Endpoint.Id)
	}

	err = Endpoint.Delete()
	if err != nil {
		t.Error(err)
	}
	err = Network.Delete()
	if err != nil {
		t.Error(err)
	}
}
func testGetEndpointByIpAddress(t *testing.T, hns HostNetworkService) {
	Network, err := createTestNetwork()
	if err != nil {
		t.Error(err)
	}

	ipConfig := &hcn.IpConfig{
		IpAddress: epIpAddress,
	}
	Endpoint := &hcn.HostComputeEndpoint{
		IpConfigurations: []hcn.IpConfig{*ipConfig},
		MacAddress:       epMacAddress,
		SchemaVersion: hcn.SchemaVersion{
			Major: 2,
			Minor: 0,
		},
	}
	Endpoint, err = Network.CreateEndpoint(Endpoint)
	if err != nil {
		t.Error(err)
	}

	endpoint, err := hns.getEndpointByIpAddress(Endpoint.IpConfigurations[0].IpAddress, Network.Name)
	if err != nil {
		t.Error(err)
	}
	if !strings.EqualFold(endpoint.hnsID, Endpoint.Id) {
		t.Errorf("%v does not match %v", endpoint.hnsID, Endpoint.Id)
	}
	if endpoint.ip != Endpoint.IpConfigurations[0].IpAddress {
		t.Errorf("%v does not match %v", endpoint.ip, Endpoint.IpConfigurations[0].IpAddress)
	}

	err = Endpoint.Delete()
	if err != nil {
		t.Error(err)
	}
	err = Network.Delete()
	if err != nil {
		t.Error(err)
	}
}
func testCreateEndpointLocal(t *testing.T, hns HostNetworkService) {
	Network, err := createTestNetwork()
	if err != nil {
		t.Error(err)
	}

	endpoint := &endpointsInfo{
		ip:         epIpAddress,
		macAddress: epMacAddress,
		isLocal:    true,
	}

	endpoint, err = hns.createEndpoint(endpoint, Network.Name)
	if err != nil {
		t.Error(err)
	}
	Endpoint, err := hcn.GetEndpointByID(endpoint.hnsID)
	if err != nil {
		t.Error(err)
	}
	if !strings.EqualFold(endpoint.hnsID, Endpoint.Id) {
		t.Errorf("%v does not match %v", endpoint.hnsID, Endpoint.Id)
	}
	if endpoint.ip != Endpoint.IpConfigurations[0].IpAddress {
		t.Errorf("%v does not match %v", endpoint.ip, Endpoint.IpConfigurations[0].IpAddress)
	}
	if endpoint.macAddress != Endpoint.MacAddress {
		t.Errorf("%v does not match %v", endpoint.macAddress, Endpoint.MacAddress)
	}

	err = Endpoint.Delete()
	if err != nil {
		t.Error(err)
	}
	err = Network.Delete()
	if err != nil {
		t.Error(err)
	}
}
func testCreateEndpointRemote(t *testing.T, hns HostNetworkService, providerAddress string) {
	Network, err := createTestNetwork()
	if err != nil {
		t.Error(err)
	}

	endpoint := &endpointsInfo{
		ip:              epIpAddressRemote,
		macAddress:      epMacAddress,
		isLocal:         false,
		providerAddress: providerAddress,
	}

	endpoint, err = hns.createEndpoint(endpoint, Network.Name)
	if err != nil {
		t.Error(err)
	}
	Endpoint, err := hcn.GetEndpointByID(endpoint.hnsID)
	if err != nil {
		t.Error(err)
	}
	if !strings.EqualFold(endpoint.hnsID, Endpoint.Id) {
		t.Errorf("%v does not match %v", endpoint.hnsID, Endpoint.Id)
	}
	if endpoint.ip != Endpoint.IpConfigurations[0].IpAddress {
		t.Errorf("%v does not match %v", endpoint.ip, Endpoint.IpConfigurations[0].IpAddress)
	}
	if endpoint.macAddress != Endpoint.MacAddress {
		t.Errorf("%v does not match %v", endpoint.macAddress, Endpoint.MacAddress)
	}
	if len(providerAddress) != 0 && endpoint.providerAddress != epPaAddress {
		t.Errorf("%v does not match %v", endpoint.providerAddress, providerAddress)
	}

	err = Endpoint.Delete()
	if err != nil {
		t.Error(err)
	}
	err = Network.Delete()
	if err != nil {
		t.Error(err)
	}
}
func testDeleteEndpoint(t *testing.T, hns HostNetworkService) {
	Network, err := createTestNetwork()
	if err != nil {
		t.Error(err)
	}

	ipConfig := &hcn.IpConfig{
		IpAddress: epIpAddress,
	}
	Endpoint := &hcn.HostComputeEndpoint{
		IpConfigurations: []hcn.IpConfig{*ipConfig},
		MacAddress:       epMacAddress,
		SchemaVersion: hcn.SchemaVersion{
			Major: 2,
			Minor: 0,
		},
	}
	Endpoint, err = Network.CreateEndpoint(Endpoint)
	if err != nil {
		t.Error(err)
	}
	err = hns.deleteEndpoint(Endpoint.Id)
	if err != nil {
		t.Error(err)
	}
	// Endpoint should no longer exist so this should fail
	Endpoint, err = hcn.GetEndpointByID(Endpoint.Id)
	if err == nil {
		t.Error(err)
	}

	err = Network.Delete()
	if err != nil {
		t.Error(err)
	}
}

func testGetLoadBalancerExisting(t *testing.T, hns HostNetworkService) {
	Network, err := createTestNetwork()
	if err != nil {
		t.Error(err)
	}

	ipConfig := &hcn.IpConfig{
		IpAddress: epIpAddress,
	}
	Endpoint := &hcn.HostComputeEndpoint{
		IpConfigurations: []hcn.IpConfig{*ipConfig},
		MacAddress:       epMacAddress,
		SchemaVersion: hcn.SchemaVersion{
			Major: 2,
			Minor: 0,
		},
	}
	Endpoint, err = Network.CreateEndpoint(Endpoint)
	if err != nil {
		t.Error(err)
	}

	Endpoints := []hcn.HostComputeEndpoint{*Endpoint}
	LoadBalancer, err := hcn.AddLoadBalancer(
		Endpoints,
		hcn.LoadBalancerFlagsNone,
		hcn.LoadBalancerPortMappingFlagsNone,
		sourceVip,
		[]string{serviceVip},
		protocol,
		internalPort,
		externalPort,
	)
	if err != nil {
		t.Error(err)
	}

	endpoint := &endpointsInfo{
		ip:    Endpoint.IpConfigurations[0].IpAddress,
		hnsID: Endpoint.Id,
	}
	endpoints := []endpointsInfo{*endpoint}
	lb, err := hns.getLoadBalancer(endpoints, loadBalancerFlags{}, sourceVip, serviceVip, protocol, internalPort, externalPort)
	if err != nil {
		t.Error(err)
	}

	if !strings.EqualFold(lb.hnsID, LoadBalancer.Id) {
		t.Errorf("%v does not match %v", lb.hnsID, LoadBalancer.Id)
	}

	err = LoadBalancer.Delete()
	if err != nil {
		t.Error(err)
	}
	err = Endpoint.Delete()
	if err != nil {
		t.Error(err)
	}
	err = Network.Delete()
	if err != nil {
		t.Error(err)
	}
}
func testGetLoadBalancerNew(t *testing.T, hns HostNetworkService) {
	Network, err := createTestNetwork()
	if err != nil {
		t.Error(err)
	}

	ipConfig := &hcn.IpConfig{
		IpAddress: epIpAddress,
	}
	Endpoint := &hcn.HostComputeEndpoint{
		IpConfigurations: []hcn.IpConfig{*ipConfig},
		MacAddress:       epMacAddress,
		SchemaVersion: hcn.SchemaVersion{
			Major: 2,
			Minor: 0,
		},
	}
	Endpoint, err = Network.CreateEndpoint(Endpoint)
	if err != nil {
		t.Error(err)
	}
	endpoint := &endpointsInfo{
		ip:    Endpoint.IpConfigurations[0].IpAddress,
		hnsID: Endpoint.Id,
	}
	endpoints := []endpointsInfo{*endpoint}
	lb, err := hns.getLoadBalancer(endpoints, loadBalancerFlags{}, sourceVip, serviceVip, protocol, internalPort, externalPort)
	if err != nil {
		t.Error(err)
	}
	LoadBalancer, err := hcn.GetLoadBalancerByID(lb.hnsID)
	if err != nil {
		t.Error(err)
	}
	if !strings.EqualFold(lb.hnsID, LoadBalancer.Id) {
		t.Errorf("%v does not match %v", lb.hnsID, LoadBalancer.Id)
	}
	err = LoadBalancer.Delete()
	if err != nil {
		t.Error(err)
	}

	err = Endpoint.Delete()
	if err != nil {
		t.Error(err)
	}
	err = Network.Delete()
	if err != nil {
		t.Error(err)
	}
}
func testDeleteLoadBalancer(t *testing.T, hns HostNetworkService) {
	Network, err := createTestNetwork()
	if err != nil {
		t.Error(err)
	}

	ipConfig := &hcn.IpConfig{
		IpAddress: epIpAddress,
	}
	Endpoint := &hcn.HostComputeEndpoint{
		IpConfigurations: []hcn.IpConfig{*ipConfig},
		MacAddress:       epMacAddress,
		SchemaVersion: hcn.SchemaVersion{
			Major: 2,
			Minor: 0,
		},
	}
	Endpoint, err = Network.CreateEndpoint(Endpoint)
	if err != nil {
		t.Error(err)
	}

	Endpoints := []hcn.HostComputeEndpoint{*Endpoint}
	LoadBalancer, err := hcn.AddLoadBalancer(
		Endpoints,
		hcn.LoadBalancerFlagsNone,
		hcn.LoadBalancerPortMappingFlagsNone,
		sourceVip,
		[]string{serviceVip},
		protocol,
		internalPort,
		externalPort,
	)
	if err != nil {
		t.Error(err)
	}
	err = hns.deleteLoadBalancer(LoadBalancer.Id)
	if err != nil {
		t.Error(err)
	}
	// Load balancer should not longer exist
	LoadBalancer, err = hcn.GetLoadBalancerByID(LoadBalancer.Id)
	if err == nil {
		t.Error(err)
	}

	err = Endpoint.Delete()
	if err != nil {
		t.Error(err)
	}
	err = Network.Delete()
	if err != nil {
		t.Error(err)
	}
}
func createTestNetwork() (*hcn.HostComputeNetwork, error) {
	network := &hcn.HostComputeNetwork{
		Type: "Overlay",
		Name: "TestOverlay",
		MacPool: hcn.MacPool{
			Ranges: []hcn.MacRange{
				{
					StartMacAddress: "00-15-5D-52-C0-00",
					EndMacAddress:   "00-15-5D-52-CF-FF",
				},
			},
		},
		Ipams: []hcn.Ipam{
			{
				Type: "Static",
				Subnets: []hcn.Subnet{
					{
						IpAddressPrefix: addressPrefix,
						Routes: []hcn.Route{
							{
								NextHop:           gatewayAddress,
								DestinationPrefix: "0.0.0.0/0",
							},
						},
					},
				},
			},
		},
		SchemaVersion: hcn.SchemaVersion{
			Major: 2,
			Minor: 0,
		},
	}

	vsid := &hcn.VsidPolicySetting{
		IsolationId: 5000,
	}
	vsidJson, err := json.Marshal(vsid)
	if err != nil {
		return nil, err
	}

	sp := &hcn.SubnetPolicy{
		Type: hcn.VSID,
	}
	sp.Settings = vsidJson

	spJson, err := json.Marshal(sp)
	if err != nil {
		return nil, err
	}

	network.Ipams[0].Subnets[0].Policies = append(network.Ipams[0].Subnets[0].Policies, spJson)

	return network.Create()
}
