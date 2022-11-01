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
	hns := hns{}
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

func TestGetEndpointByID(t *testing.T) {
	hns := hns{}
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
	hns := hns{}
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

	endpoint, err = hns.getEndpointByName(Endpoint.Name)
	if err != nil {
		t.Error(err)
	}
	diff := cmp.Diff(endpoint, Endpoint)
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
	hns := hns{}
	Network := mustTestNetwork(t)

	endpoint := &endpointsInfo{
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
	hns := hns{}
	Network := mustTestNetwork(t)
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
	hns := hns{}
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
	hns := hns{}
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
	endpoint := &endpointsInfo{
		ip:    Endpoint.IpConfigurations[0].IpAddress,
		hnsID: Endpoint.Id,
	}
	endpoints := []endpointsInfo{*endpoint}
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
	hns := hns{}
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
	endpoint := &endpointsInfo{
		ip:    Endpoint.IpConfigurations[0].IpAddress,
		hnsID: Endpoint.Id,
	}
	endpoints := []endpointsInfo{*endpoint}
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
	hns := hns{}
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
