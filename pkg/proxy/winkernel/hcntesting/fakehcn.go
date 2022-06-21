//go:build windows
// +build windows

/*
Copyright 2022 The Kubernetes Authors.

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

package hcntesting

import (
	"encoding/json"
	"errors"

	"github.com/Microsoft/hcsshim/hcn"
	"k8s.io/klog/v2"
)

const (
	guid                        = "123ABC"
	destinationPrefix           = "192.168.2.0/24"
	isolationId                 = 4096
	providerAddress             = "10.0.0.3"
	distributedRouterMacAddress = "00-11-22-33-44-55"
	macRangeStartAddress        = "00-15-5D-52-C0-00"
	macRangeEndAddress          = "00-15-5D-52-CF-FF"
	ipAddressPrefix             = "192.168.1.0/24"
	subnetNextHop               = "192.168.1.1"
	subnetDestinationPrefix     = "0.0.0.0/0"
)

//the fakeHCN saves the created endpoints and loadbalancers in slices to be able to work with them easier
type FakeHCN struct {
	Endpoints     []*hcn.HostComputeEndpoint
	Loadbalancers []*hcn.HostComputeLoadBalancer
}

func (HCN *FakeHCN) GetNetworkByName(networkName string) (*hcn.HostComputeNetwork, error) {
	policysettings := &hcn.RemoteSubnetRoutePolicySetting{
		DestinationPrefix:           destinationPrefix,
		IsolationId:                 isolationId,
		ProviderAddress:             providerAddress,
		DistributedRouterMacAddress: distributedRouterMacAddress,
	}

	jsonsettings, err := json.Marshal(policysettings)
	if err != nil {
		klog.ErrorS(err, "failed to encode policy settings")
	}
	policy := &hcn.NetworkPolicy{
		Type:     hcn.RemoteSubnetRoute,
		Settings: jsonsettings,
	}

	var policies []hcn.NetworkPolicy
	policies = append(policies, *policy)

	return &hcn.HostComputeNetwork{
		Id:   guid,
		Name: networkName,
		Type: "overlay",
		MacPool: hcn.MacPool{
			Ranges: []hcn.MacRange{
				{
					StartMacAddress: macRangeStartAddress,
					EndMacAddress:   macRangeEndAddress,
				},
			},
		},
		Ipams: []hcn.Ipam{
			{
				Type: "Static",
				Subnets: []hcn.Subnet{
					{
						IpAddressPrefix: ipAddressPrefix,
						Routes: []hcn.Route{
							{
								NextHop:           subnetNextHop,
								DestinationPrefix: subnetDestinationPrefix,
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
		Policies: policies,
	}, nil
}

func (HCN *FakeHCN) ListEndpointsOfNetwork(networkId string) ([]hcn.HostComputeEndpoint, error) {
	var endpoints []hcn.HostComputeEndpoint
	for _, ep := range HCN.Endpoints {
		if ep.HostComputeNetwork == networkId {
			endpoints = append(endpoints, *ep)
		}
	}
	return endpoints, nil
}

func (HCN *FakeHCN) GetEndpointByID(endpointId string) (*hcn.HostComputeEndpoint, error) {
	endpoint := &hcn.HostComputeEndpoint{}
	for _, ep := range HCN.Endpoints {
		if ep.Id == endpointId {
			endpoint = ep
			break
		}
	}
	if endpoint.Id == "" {
		err := errors.New("No endpoint matches this ID")
		return nil, err
	}
	return endpoint, nil
}

func (HCN *FakeHCN) ListEndpoints() ([]hcn.HostComputeEndpoint, error) {

	var endpoints []hcn.HostComputeEndpoint
	for _, ep := range HCN.Endpoints {
		endpoints = append(endpoints, *ep)
	}
	return endpoints, nil
}

func (HCN *FakeHCN) GetEndpointByName(endpointName string) (*hcn.HostComputeEndpoint, error) {
	endpoint := &hcn.HostComputeEndpoint{}
	for _, ep := range HCN.Endpoints {
		if ep.Name == endpointName {
			endpoint = ep
			break
		}
	}
	if endpoint.Name == "" {
		err := errors.New("No endpoint matches this NAME")
		return nil, err
	}
	return endpoint, nil
}

func (HCN *FakeHCN) ListLoadBalancers() ([]hcn.HostComputeLoadBalancer, error) {
	var loadbalancers []hcn.HostComputeLoadBalancer
	for _, lb := range HCN.Loadbalancers {
		loadbalancers = append(loadbalancers, *lb)
	}
	return loadbalancers, nil
}

func (HCN *FakeHCN) GetLoadBalancerByID(loadBalancerId string) (*hcn.HostComputeLoadBalancer, error) {
	loadbalancer := &hcn.HostComputeLoadBalancer{}
	for _, lb := range HCN.Loadbalancers {
		if lb.Id == loadBalancerId {
			loadbalancer = lb
			break
		}
	}
	if loadbalancer.Id == "" {
		err := errors.New("No loadBalancer matches this ID")
		return nil, err
	}
	return loadbalancer, nil
}

func (HCN *FakeHCN) CreateEndpoint(endpoint *hcn.HostComputeEndpoint, network *hcn.HostComputeNetwork) (*hcn.HostComputeEndpoint, error) {
	newEndpoint := &hcn.HostComputeEndpoint{
		Id:                   guid,
		Name:                 endpoint.Name,
		HostComputeNetwork:   guid,
		IpConfigurations:     endpoint.IpConfigurations,
		MacAddress:           endpoint.MacAddress,
		Flags:                hcn.EndpointFlagsNone,
		SchemaVersion:        endpoint.SchemaVersion,
		Policies:             endpoint.Policies,
		HostComputeNamespace: endpoint.HostComputeNamespace,
		Dns:                  endpoint.Dns,
		Routes:               endpoint.Routes,
		Health:               endpoint.Health,
	}

	HCN.Endpoints = append(HCN.Endpoints, newEndpoint)

	return newEndpoint, nil
}

func (HCN *FakeHCN) CreateRemoteEndpoint(endpoint *hcn.HostComputeEndpoint, network *hcn.HostComputeNetwork) (*hcn.HostComputeEndpoint, error) {
	newEndpoint := &hcn.HostComputeEndpoint{
		Id:                   guid,
		Name:                 endpoint.Name,
		HostComputeNetwork:   guid,
		IpConfigurations:     endpoint.IpConfigurations,
		MacAddress:           endpoint.MacAddress,
		Flags:                hcn.EndpointFlagsRemoteEndpoint | endpoint.Flags,
		SchemaVersion:        endpoint.SchemaVersion,
		Policies:             endpoint.Policies,
		HostComputeNamespace: endpoint.HostComputeNamespace,
		Dns:                  endpoint.Dns,
		Routes:               endpoint.Routes,
		Health:               endpoint.Health,
	}

	HCN.Endpoints = append(HCN.Endpoints, newEndpoint)

	return newEndpoint, nil
}

func (HCN *FakeHCN) CreateLoadBalancer(loadbalancer *hcn.HostComputeLoadBalancer) (*hcn.HostComputeLoadBalancer, error) {
	newLoadBalancer := &hcn.HostComputeLoadBalancer{
		Id:                   guid,
		HostComputeEndpoints: loadbalancer.HostComputeEndpoints,
		SourceVIP:            loadbalancer.SourceVIP,
		Flags:                loadbalancer.Flags,
		FrontendVIPs:         loadbalancer.FrontendVIPs,
		PortMappings:         loadbalancer.PortMappings,
		SchemaVersion:        loadbalancer.SchemaVersion,
	}

	HCN.Loadbalancers = append(HCN.Loadbalancers, newLoadBalancer)

	return newLoadBalancer, nil
}

func (HCN *FakeHCN) DeleteLoadBalancer(loadbalancer *hcn.HostComputeLoadBalancer) error {
	var i int

	for _, lb := range HCN.Loadbalancers {
		i++
		if lb.Id == loadbalancer.Id {
			break
		}
	}

	i--

	if len(HCN.Loadbalancers) != 0 { //here we actually delete the load balancer from fakehcn memory
		copy(HCN.Loadbalancers[i:], HCN.Loadbalancers[i+1:])
		HCN.Loadbalancers[len(HCN.Loadbalancers)-1] = nil
		HCN.Loadbalancers = HCN.Loadbalancers[:len(HCN.Loadbalancers)-1]
	}

	return nil
}

func (HCN *FakeHCN) DeleteEndpoint(endpoint *hcn.HostComputeEndpoint) error {
	var i int

	for _, ep := range HCN.Endpoints {
		i++
		if ep.Id == endpoint.Id {
			break
		}
	}

	i--

	if len(HCN.Endpoints) != 0 { //here we actually delete the endpoint from fakehcn memory
		copy(HCN.Endpoints[i:], HCN.Endpoints[i+1:])
		HCN.Endpoints[len(HCN.Endpoints)-1] = nil
		HCN.Endpoints = HCN.Endpoints[:len(HCN.Endpoints)-1]
	}

	return nil
}
