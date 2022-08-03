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
	"encoding/json"

	"github.com/Microsoft/hcsshim/hcn"
	"github.com/stretchr/testify/assert"

	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
)

const (
	sourceVip         = "192.168.1.2"
	serviceVip        = "11.0.0.1"
	addressPrefix     = "192.168.1.0/24"
	gatewayAddress    = "192.168.1.1"
	epMacAddress      = "00-11-22-33-44-55"
	epIpAddress       = "192.168.1.3"
	epIpAddressRemote = "192.168.2.3"
	epPaAddress       = "10.0.0.3"
	protocol          = 6
	internalPort      = 80
	externalPort      = 32440
)

func TestGetNetworkByName(t *testing.T) {
	hns := hns{
		hcnUtils: newMockHCN(1),
	}
	Network := mustTestNetwork(t, hns)

	network, err := hns.getNetworkByName(Network.Name)
	if err != nil {
		t.Error(err)
	}

	if !strings.EqualFold(network.id, Network.Id) {
		t.Errorf("%v does not match %v", network.id, Network.Id)
	}
	err = hns.hcnUtils.deleteNetwork(Network)
	if err != nil {
		t.Error(err)
	}
}

func TestGetEndpointByID(t *testing.T) {
	hns := hns{
		hcnUtils: newMockHCN(1),
	}
	Network := mustTestNetwork(t, hns)

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

	Endpoint, err := hns.hcnUtils.createEndpoint(Endpoint, Network)
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

	err = hns.hcnUtils.deleteEndpoint(Endpoint)
	if err != nil {
		t.Error(err)
	}
	err = hns.hcnUtils.deleteNetwork(Network)
	if err != nil {
		t.Error(err)
	}
}

func TestGetEndpointByIpAddressAndName(t *testing.T) {
	hns := hns{
		hcnUtils: newMockHCN(1),
	}
	Network := mustTestNetwork(t, hns)

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
	endpoint, err := hns.hcnUtils.createEndpoint(Endpoint, Network)
	if err != nil {
		t.Error(err)
	}

	endpointInfoActual, err := hns.getEndpointByIpAddress(Endpoint.IpConfigurations[0].IpAddress, Network.Name)
	assert.NoError(t, err, "Error in getEndpointByIpAddress function.")

	endpointInfoExpected := &endpointsInfo{
		ip:         Endpoint.IpConfigurations[0].IpAddress,
		isLocal:    uint32(endpoint.Flags&hcn.EndpointFlagsRemoteEndpoint) == 0, //TODO: Change isLocal to isRemote
		macAddress: endpoint.MacAddress,
		hnsID:      endpoint.Id,
		hns:        hns,
	}

	diff := cmp.Diff(endpointInfoExpected, endpointInfoActual)
	assert.Equal(t, diff, "", "getEndpointByIpAddress(%s) returned a different endpoint. Diff: %s ", Endpoint.Name, diff)

	endpointInfoActual, err = hns.getEndpointByName(endpoint.Name)
	assert.NoError(t, err, "Error in getEndpointByName function.")

	diff = cmp.Diff(endpointInfoExpected, endpointInfoActual)
	assert.Equal(t, diff, "", "getEndpointByName(%s) returned a different endpoint. Diff: %s ", Endpoint.Name, diff)

	err = hns.hcnUtils.deleteEndpoint(Endpoint)
	if err != nil {
		t.Error(err)
	}
	err = hns.hcnUtils.deleteNetwork(Network)
	if err != nil {
		t.Error(err)
	}
}

func TestCreateEndpointLocal(t *testing.T) {
	hns := hns{
		hcnUtils: newMockHCN(1),
	}
	Network := mustTestNetwork(t, hns)

	endpoint := &endpointsInfo{
		ip:         epIpAddress,
		macAddress: epMacAddress,
		isLocal:    true,
	}

	endpoint, err := hns.createEndpoint(endpoint, Network.Name)
	if err != nil {
		t.Error(err)
	}
	Endpoint, err := hns.hcnUtils.getEndpointByID(endpoint.hnsID)
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

	err = hns.hcnUtils.deleteEndpoint(Endpoint)
	if err != nil {
		t.Error(err)
	}
	err = hns.hcnUtils.deleteNetwork(Network)
	if err != nil {
		t.Error(err)
	}
}

func TestCreateEndpointRemote(t *testing.T) {
	hns := hns{
		hcnUtils: newMockHCN(1),
	}
	Network := mustTestNetwork(t, hns)
	providerAddress := epPaAddress

	endpoint := &endpointsInfo{
		ip:              epIpAddressRemote,
		macAddress:      epMacAddress,
		isLocal:         false,
		providerAddress: providerAddress,
	}

	endpoint, err := hns.createEndpoint(endpoint, Network.Name)
	if err != nil {
		t.Error(err)
	}
	Endpoint, err := hns.hcnUtils.getEndpointByID(endpoint.hnsID)
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

	err = hns.hcnUtils.deleteEndpoint(Endpoint)
	if err != nil {
		t.Error(err)
	}
	err = hns.hcnUtils.deleteNetwork(Network)
	if err != nil {
		t.Error(err)
	}
}

func TestDeleteEndpoint(t *testing.T) {
	hns := hns{
		hcnUtils: newMockHCN(1),
	}
	Network := mustTestNetwork(t, hns)

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
	Endpoint, err := hns.hcnUtils.createEndpoint(Endpoint, Network)
	if err != nil {
		t.Error(err)
	}
	err = hns.deleteEndpoint(Endpoint.Id)
	if err != nil {
		t.Error(err)
	}
	// Endpoint should no longer exist so this should fail
	Endpoint, err = hns.hcnUtils.getEndpointByID(Endpoint.Id)
	if err == nil {
		t.Error(err)
	}

	err = hns.hcnUtils.deleteEndpoint(Endpoint)
	if err != nil {
		t.Error(err)
	}
	err = hns.hcnUtils.deleteNetwork(Network)
	if err != nil {
		t.Error(err)
	}
}

func TestGetLoadBalancerExisting(t *testing.T) {
	hns := hns{
		hcnUtils: newMockHCN(1),
	}
	Network := mustTestNetwork(t, hns)
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
	Endpoint, err := hns.hcnUtils.createEndpoint(Endpoint, Network)
	if err != nil {
		t.Error(err)
	}

	Endpoints := []hcn.HostComputeEndpoint{*Endpoint}
	LoadBalancer, err := mockAddLoadBalancer(hns,
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
	// We populate this to ensure we test for getting existing load balancer
	id := loadBalancerIdentifier{protocol: protocol, internalPort: internalPort, externalPort: externalPort, vip: serviceVip, endpointsCount: len(Endpoints)}
	lbs[id] = &loadBalancerInfo{hnsID: LoadBalancer.Id}

	endpoint := &endpointsInfo{
		ip:    Endpoint.IpConfigurations[0].IpAddress,
		hnsID: Endpoint.Id,
	}
	endpoints := []endpointsInfo{*endpoint}
	lb, err := hns.getLoadBalancer(endpoints, loadBalancerFlags{}, sourceVip, serviceVip, protocol, internalPort, externalPort, lbs)

	if err != nil {
		t.Error(err)
	}

	if !strings.EqualFold(lb.hnsID, LoadBalancer.Id) {
		t.Errorf("%v does not match %v", lb.hnsID, LoadBalancer.Id)
	}

	err = hns.hcnUtils.deleteLoadBalancer(LoadBalancer)
	if err != nil {
		t.Error(err)
	}
	err = hns.hcnUtils.deleteEndpoint(Endpoint)
	if err != nil {
		t.Error(err)
	}
	err = hns.hcnUtils.deleteNetwork(Network)
	if err != nil {
		t.Error(err)
	}
}

func TestGetLoadBalancerNew(t *testing.T) {
	hns := hns{
		hcnUtils: newMockHCN(1),
	}
	Network := mustTestNetwork(t, hns)
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
	Endpoint, err := hns.hcnUtils.createEndpoint(Endpoint, Network)
	if err != nil {
		t.Error(err)
	}
	endpoint := &endpointsInfo{
		ip:    Endpoint.IpConfigurations[0].IpAddress,
		hnsID: Endpoint.Id,
	}
	endpoints := []endpointsInfo{*endpoint}
	lb, err := hns.getLoadBalancer(endpoints, loadBalancerFlags{}, sourceVip, serviceVip, protocol, internalPort, externalPort, lbs)
	if err != nil {
		t.Error(err)
	}
	LoadBalancer, err := hns.hcnUtils.getLoadBalancerByID(lb.hnsID)
	if err != nil {
		t.Error(err)
	}
	if !strings.EqualFold(lb.hnsID, LoadBalancer.Id) {
		t.Errorf("%v does not match %v", lb.hnsID, LoadBalancer.Id)
	}
	err = hns.hcnUtils.deleteLoadBalancer(LoadBalancer)
	if err != nil {
		t.Error(err)
	}
	err = hns.hcnUtils.deleteEndpoint(Endpoint)
	if err != nil {
		t.Error(err)
	}
	err = hns.hcnUtils.deleteNetwork(Network)
	if err != nil {
		t.Error(err)
	}
}

func TestDeleteLoadBalancer(t *testing.T) {
	hns := hns{
		hcnUtils: newMockHCN(1),
	}
	Network := mustTestNetwork(t, hns)

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
	Endpoint, err := hns.hcnUtils.createEndpoint(Endpoint, Network)
	if err != nil {
		t.Error(err)
	}

	Endpoints := []hcn.HostComputeEndpoint{*Endpoint}
	LoadBalancer, err := mockAddLoadBalancer(
		hns,
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
	LoadBalancer, err = hns.hcnUtils.getLoadBalancerByID(LoadBalancer.Id)
	if err == nil {
		t.Error(err)
	}

	err = hns.hcnUtils.deleteLoadBalancer(LoadBalancer)
	if err != nil {
		t.Error(err)
	}
	err = hns.hcnUtils.deleteEndpoint(Endpoint)
	if err != nil {
		t.Error(err)
	}
	err = hns.hcnUtils.deleteNetwork(Network)
	if err != nil {
		t.Error(err)
	}
}

func mustTestNetwork(t *testing.T, hns hns) *hcn.HostComputeNetwork {
	network, err := createTestNetwork(hns)
	if err != nil {
		t.Fatalf("cannot create test network: %v", err)
	}
	if network == nil {
		t.Fatal("test network was nil without error")
	}
	return network
}

func createTestNetwork(hns hns) (*hcn.HostComputeNetwork, error) {
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

	return hns.hcnUtils.createNetwork(network)
}

func mockAddLoadBalancer(hns hns, endpoints []hcn.HostComputeEndpoint, flags hcn.LoadBalancerFlags, portMappingFlags hcn.LoadBalancerPortMappingFlags, sourceVIP string, frontendVIPs []string, protocol uint16, internalPort uint16, externalPort uint16) (*hcn.HostComputeLoadBalancer, error) {
	loadBalancer := &hcn.HostComputeLoadBalancer{
		SourceVIP: sourceVIP,
		PortMappings: []hcn.LoadBalancerPortMapping{
			{
				Protocol:     uint32(protocol),
				InternalPort: internalPort,
				ExternalPort: externalPort,
				Flags:        portMappingFlags,
			},
		},
		FrontendVIPs: frontendVIPs,
		SchemaVersion: hcn.SchemaVersion{
			Major: 2,
			Minor: 0,
		},
		Flags: flags,
	}

	for _, endpoint := range endpoints {
		loadBalancer.HostComputeEndpoints = append(loadBalancer.HostComputeEndpoints, endpoint.Id)
	}

	return hns.hcnUtils.createLoadBalancer(loadBalancer)
}
