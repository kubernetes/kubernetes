//go:build windows

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
	"strings"
	"testing"

	"github.com/Microsoft/hnslib/hcn"
	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"
)

const (
	sourceVip         = "192.168.1.2"
	serviceVip        = "11.0.0.1"
	addressPrefix     = "192.168.1.0/24"
	gatewayAddress    = "192.168.1.1"
	epMacAddress      = "00-11-22-33-44-55"
	epIpAddress       = "192.168.1.3"
	epIpv6Address     = "192::3"
	epIpAddressB      = "192.168.1.4"
	epIpAddressRemote = "192.168.2.3"
	epIpAddressLocal1 = "192.168.4.4"
	epIpAddressLocal2 = "192.168.4.5"
	epPaAddress       = "10.0.0.3"
	protocol          = 6
	internalPort      = 80
	externalPort      = 32440
)

func TestGetNetworkByName(t *testing.T) {
	// TODO: remove skip once the test has been fixed.
	t.Skip("Skipping failing test on Windows.")
	hns := hns{hcn: newHcnImpl()}
	Network := mustTestNetwork(t)

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

func TestGetAllEndpointsByNetwork(t *testing.T) {
	// TODO: remove skip once the test has been fixed.
	t.Skip("Skipping failing test on Windows.")
	hns := hns{hcn: newHcnImpl()}
	Network := mustTestNetwork(t)

	ipv4Config := &hcn.IpConfig{
		IpAddress: epIpAddress,
	}
	ipv6Config := &hcn.IpConfig{
		IpAddress: epIpv6Address,
	}
	Endpoint := &hcn.HostComputeEndpoint{
		IpConfigurations: []hcn.IpConfig{*ipv4Config, *ipv6Config},
		MacAddress:       epMacAddress,
		SchemaVersion: hcn.SchemaVersion{
			Major: 2,
			Minor: 0,
		},
	}
	Endpoint, err := Network.CreateEndpoint(Endpoint)
	if err != nil {
		t.Error(err)
	}

	mapEndpointsInfo, err := hns.getAllEndpointsByNetwork(Network.Name)
	if err != nil {
		t.Error(err)
	}
	endpointIpv4, ipv4EpPresent := mapEndpointsInfo[ipv4Config.IpAddress]
	assert.True(t, ipv4EpPresent, "IPV4 endpoint is missing in Dualstack mode")
	assert.Equal(t, endpointIpv4.ip, epIpAddress, "IPV4 IP is missing in Dualstack mode")

	endpointIpv6, ipv6EpPresent := mapEndpointsInfo[ipv6Config.IpAddress]
	assert.True(t, ipv6EpPresent, "IPV6 endpoint is missing in Dualstack mode")
	assert.Equal(t, endpointIpv6.ip, epIpv6Address, "IPV6 IP is missing in Dualstack mode")

	err = Endpoint.Delete()
	if err != nil {
		t.Error(err)
	}
	err = Network.Delete()
	if err != nil {
		t.Error(err)
	}
}

func TestGetAllEndpointsByNetworkWithDupEP(t *testing.T) {
	hcnMock := getHcnMock("L2Bridge")
	hns := hns{hcn: hcnMock}

	ipv4Config := &hcn.IpConfig{
		IpAddress: epIpAddress,
	}
	ipv6Config := &hcn.IpConfig{
		IpAddress: epIpv6Address,
	}
	remoteEndpoint := &hcn.HostComputeEndpoint{
		IpConfigurations: []hcn.IpConfig{*ipv4Config, *ipv6Config},
		MacAddress:       epMacAddress,
		SchemaVersion: hcn.SchemaVersion{
			Major: 2,
			Minor: 0,
		},
		Flags: hcn.EndpointFlagsRemoteEndpoint,
	}
	Network, _ := hcnMock.GetNetworkByName(testNetwork)
	remoteEndpoint, err := hns.hcn.CreateEndpoint(Network, remoteEndpoint)
	if err != nil {
		t.Error(err)
	}

	// Create a duplicate local endpoint with the same IP address
	dupLocalEndpoint := &hcn.HostComputeEndpoint{
		IpConfigurations: []hcn.IpConfig{*ipv4Config, *ipv6Config},
		MacAddress:       epMacAddress,
		SchemaVersion: hcn.SchemaVersion{
			Major: 2,
			Minor: 0,
		},
	}

	dupLocalEndpoint, err = hns.hcn.CreateEndpoint(Network, dupLocalEndpoint)
	if err != nil {
		t.Error(err)
	}

	mapEndpointsInfo, err := hns.getAllEndpointsByNetwork(Network.Name)
	if err != nil {
		t.Error(err)
	}
	endpointIpv4, ipv4EpPresent := mapEndpointsInfo[ipv4Config.IpAddress]
	assert.True(t, ipv4EpPresent, "IPV4 endpoint is missing in Dualstack mode")
	assert.Equal(t, endpointIpv4.ip, epIpAddress, "IPV4 IP is missing in Dualstack mode")
	assert.Equal(t, endpointIpv4.hnsID, remoteEndpoint.Id, "HNS ID is not matching with remote endpoint")

	endpointIpv6, ipv6EpPresent := mapEndpointsInfo[ipv6Config.IpAddress]
	assert.True(t, ipv6EpPresent, "IPV6 endpoint is missing in Dualstack mode")
	assert.Equal(t, endpointIpv6.ip, epIpv6Address, "IPV6 IP is missing in Dualstack mode")
	assert.Equal(t, endpointIpv6.hnsID, remoteEndpoint.Id, "HNS ID is not matching with remote endpoint")

	err = hns.hcn.DeleteEndpoint(remoteEndpoint)
	if err != nil {
		t.Error(err)
	}
	err = hns.hcn.DeleteEndpoint(dupLocalEndpoint)
	if err != nil {
		t.Error(err)
	}
}

func TestGetEndpointByID(t *testing.T) {
	// TODO: remove skip once the test has been fixed.
	t.Skip("Skipping failing test on Windows.")
	hns := hns{hcn: newHcnImpl()}
	Network := mustTestNetwork(t)

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

	Endpoint, err := Network.CreateEndpoint(Endpoint)
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

func TestGetEndpointByIpAddressAndName(t *testing.T) {
	// TODO: remove skip once the test has been fixed.
	t.Skip("Skipping failing test on Windows.")
	hns := hns{hcn: newHcnImpl()}
	Network := mustTestNetwork(t)

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
	Endpoint, err := Network.CreateEndpoint(Endpoint)
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

	endpoint2, err := hns.getEndpointByName(Endpoint.Name)
	if err != nil {
		t.Error(err)
	}
	diff := cmp.Diff(endpoint, endpoint2)
	if diff != "" {
		t.Errorf("getEndpointByName(%s) returned a different endpoint. Diff: %s ", Endpoint.Name, diff)
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

func TestCreateEndpointLocal(t *testing.T) {
	// TODO: remove skip once the test has been fixed.
	t.Skip("Skipping failing test on Windows.")
	hns := hns{hcn: newHcnImpl()}
	Network := mustTestNetwork(t)

	endpoint := &endpointInfo{
		ip:         epIpAddress,
		macAddress: epMacAddress,
		isLocal:    true,
	}

	endpoint, err := hns.createEndpoint(endpoint, Network.Name)
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

func TestCreateEndpointRemote(t *testing.T) {
	// TODO: remove skip once the test has been fixed.
	t.Skip("Skipping failing test on Windows.")
	hns := hns{hcn: newHcnImpl()}
	Network := mustTestNetwork(t)
	providerAddress := epPaAddress

	endpoint := &endpointInfo{
		ip:              epIpAddressRemote,
		macAddress:      epMacAddress,
		isLocal:         false,
		providerAddress: providerAddress,
	}

	endpoint, err := hns.createEndpoint(endpoint, Network.Name)
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

func TestDeleteEndpoint(t *testing.T) {
	// TODO: remove skip once the test has been fixed.
	t.Skip("Skipping failing test on Windows.")
	hns := hns{hcn: newHcnImpl()}
	Network := mustTestNetwork(t)

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
	Endpoint, err := Network.CreateEndpoint(Endpoint)
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

func TestGetLoadBalancerExisting(t *testing.T) {
	// TODO: remove skip once the test has been fixed.
	t.Skip("Skipping failing test on Windows.")
	hns := hns{hcn: newHcnImpl()}
	Network := mustTestNetwork(t)
	lbs := make(map[loadBalancerIdentifier]*(loadBalancerInfo))

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
	Endpoint, err := Network.CreateEndpoint(Endpoint)
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
	endpoint := &endpointInfo{
		ip:    Endpoint.IpConfigurations[0].IpAddress,
		hnsID: Endpoint.Id,
	}
	endpoints := []endpointInfo{*endpoint}
	hash, err := hashEndpoints(endpoints)
	if err != nil {
		t.Error(err)
	}

	// We populate this to ensure we test for getting existing load balancer
	id := loadBalancerIdentifier{protocol: protocol, internalPort: internalPort, externalPort: externalPort, vip: serviceVip, endpointsHash: hash}
	lbs[id] = &loadBalancerInfo{hnsID: LoadBalancer.Id}

	lb, err := hns.getLoadBalancer(endpoints, loadBalancerFlags{}, sourceVip, serviceVip, protocol, internalPort, externalPort, lbs)

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

func TestGetLoadBalancerNew(t *testing.T) {
	// TODO: remove skip once the test has been fixed.
	t.Skip("Skipping failing test on Windows.")
	hns := hns{hcn: newHcnImpl()}
	Network := mustTestNetwork(t)
	// We keep this empty to ensure we test for new load balancer creation.
	lbs := make(map[loadBalancerIdentifier]*(loadBalancerInfo))

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
	Endpoint, err := Network.CreateEndpoint(Endpoint)
	if err != nil {
		t.Error(err)
	}
	endpoint := &endpointInfo{
		ip:    Endpoint.IpConfigurations[0].IpAddress,
		hnsID: Endpoint.Id,
	}
	endpoints := []endpointInfo{*endpoint}
	lb, err := hns.getLoadBalancer(endpoints, loadBalancerFlags{}, sourceVip, serviceVip, protocol, internalPort, externalPort, lbs)
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

func TestDeleteLoadBalancer(t *testing.T) {
	// TODO: remove skip once the test has been fixed.
	t.Skip("Skipping failing test on Windows.")
	hns := hns{hcn: newHcnImpl()}
	Network := mustTestNetwork(t)

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
	Endpoint, err := Network.CreateEndpoint(Endpoint)
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

func mustTestNetwork(t *testing.T) *hcn.HostComputeNetwork {
	network, err := createTestNetwork()
	if err != nil {
		t.Fatalf("cannot create test network: %v", err)
	}
	if network == nil {
		t.Fatal("test network was nil without error")
	}
	return network
}

func TestHashEndpoints(t *testing.T) {
	// TODO: remove skip once the test has been fixed.
	t.Skip("Skipping failing test on Windows.")
	Network := mustTestNetwork(t)
	// Create endpoint A
	ipConfigA := &hcn.IpConfig{
		IpAddress: epIpAddress,
	}
	endpointASpec := &hcn.HostComputeEndpoint{
		IpConfigurations: []hcn.IpConfig{*ipConfigA},
		MacAddress:       epMacAddress,
		SchemaVersion: hcn.SchemaVersion{
			Major: 2,
			Minor: 0,
		},
	}
	endpointA, err := Network.CreateEndpoint(endpointASpec)
	if err != nil {
		t.Error(err)
	}
	endpointInfoA := &endpointInfo{
		ip:    endpointA.IpConfigurations[0].IpAddress,
		hnsID: endpointA.Id,
	}
	// Create Endpoint B
	ipConfigB := &hcn.IpConfig{
		IpAddress: epIpAddressB,
	}
	endpointBSpec := &hcn.HostComputeEndpoint{
		IpConfigurations: []hcn.IpConfig{*ipConfigB},
		MacAddress:       epMacAddress,
		SchemaVersion: hcn.SchemaVersion{
			Major: 2,
			Minor: 0,
		},
	}
	endpointB, err := Network.CreateEndpoint(endpointBSpec)
	if err != nil {
		t.Error(err)
	}
	endpointInfoB := &endpointInfo{
		ip:    endpointB.IpConfigurations[0].IpAddress,
		hnsID: endpointB.Id,
	}
	endpoints := []endpointInfo{*endpointInfoA, *endpointInfoB}
	endpointsReverse := []endpointInfo{*endpointInfoB, *endpointInfoA}
	h1, err := hashEndpoints(endpoints)
	if err != nil {
		t.Error(err)
	} else if len(h1) < 1 {
		t.Error("HashEndpoints failed for endpoints", endpoints)
	}

	h2, err := hashEndpoints(endpointsReverse)
	if err != nil {
		t.Error(err)
	}
	if h1 != h2 {
		t.Errorf("%x does not match %x", h1, h2)
	}

	// Clean up
	err = endpointA.Delete()
	if err != nil {
		t.Error(err)
	}
	err = endpointB.Delete()
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
		Type: NETWORK_TYPE_OVERLAY,
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
