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
	"errors"
	"strconv"

	"github.com/Microsoft/hcsshim/hcn"
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
	lbIDPrefix                  = "MOCK-LB-ID-"
	epIDPrefix                  = "MOCK-EP-ID-"
)

// mockHCN saves the created endpoints and loadbalancers in slices to be able to work with them easier
type mockHCN struct {
	lbIndex         int // Loadbalancer index
	loadBalancerIDs []string
	epIDIndex       int
	endpointIDs     []string
	hnsNetworkName  string
	networks        map[string]*hcn.HostComputeNetwork
	Endpoints       []*hcn.HostComputeEndpoint
	Loadbalancers   []*hcn.HostComputeLoadBalancer
}

func newMockHCN(idCount int) *mockHCN {
	var lbIDs, epIDs []string
	for i := 0; i < idCount; i++ {
		lbIDs = append(lbIDs, lbIDPrefix+strconv.Itoa(i))
		epIDs = append(epIDs, epIDPrefix+strconv.Itoa(i))
	}
	return &mockHCN{
		lbIndex:         0,
		loadBalancerIDs: lbIDs,
		endpointIDs:     epIDs,
		hnsNetworkName:  guid,
		networks:        make(map[string]*hcn.HostComputeNetwork),
	}
}

func (hcnObj *mockHCN) createNetwork(network *hcn.HostComputeNetwork) (*hcn.HostComputeNetwork, error) {
	network.Id = network.Name
	hcnObj.networks[network.Name] = network
	return network, nil
}

func (hcnObj *mockHCN) deleteNetwork(network *hcn.HostComputeNetwork) error {
	if _, ok := hcnObj.networks[network.Name]; ok {
		delete(hcnObj.networks, network.Name)
		return nil
	}
	return errors.New("network not found")
}

func (hcnObj *mockHCN) getNetworkByName(networkName string) (*hcn.HostComputeNetwork, error) {
	if network, ok := hcnObj.networks[networkName]; ok {
		return network, nil
	}

	return nil, errors.New("network not found")
}

func (hcnObj *mockHCN) listEndpointsOfNetwork(networkId string) ([]hcn.HostComputeEndpoint, error) {
	var endpoints []hcn.HostComputeEndpoint
	for _, ep := range hcnObj.Endpoints {
		if ep.HostComputeNetwork == networkId {
			endpoints = append(endpoints, *ep)
		}
	}
	return endpoints, nil
}

func (hcnObj *mockHCN) getEndpointByID(endpointId string) (*hcn.HostComputeEndpoint, error) {
	endpoint := &hcn.HostComputeEndpoint{}
	for _, ep := range hcnObj.Endpoints {
		if ep.Id == endpointId {
			endpoint = ep
			break
		}
	}
	if endpoint.Id == "" {
		err := errors.New("no endpoint matches this ID")
		return nil, err
	}
	return endpoint, nil
}

func (hcnObj *mockHCN) listEndpoints() ([]hcn.HostComputeEndpoint, error) {

	var endpoints []hcn.HostComputeEndpoint
	for _, ep := range hcnObj.Endpoints {
		endpoints = append(endpoints, *ep)
	}
	return endpoints, nil
}

func (hcnObj *mockHCN) getEndpointByName(endpointName string) (*hcn.HostComputeEndpoint, error) {
	endpoint := &hcn.HostComputeEndpoint{}
	for _, ep := range hcnObj.Endpoints {
		if ep.Name == endpointName {
			endpoint = ep
			break
		}
	}
	if endpoint.Name == "" {
		err := errors.New("no endpoint matches this NAME")
		return nil, err
	}
	return endpoint, nil
}

func (hcnObj *mockHCN) listLoadBalancers() ([]hcn.HostComputeLoadBalancer, error) {
	var loadbalancers []hcn.HostComputeLoadBalancer
	for _, lb := range hcnObj.Loadbalancers {
		loadbalancers = append(loadbalancers, *lb)
	}
	return loadbalancers, nil
}

func (hcnObj *mockHCN) getLoadBalancerByID(loadBalancerId string) (*hcn.HostComputeLoadBalancer, error) {
	loadbalancer := &hcn.HostComputeLoadBalancer{}
	for _, lb := range hcnObj.Loadbalancers {
		if lb.Id == loadBalancerId {
			loadbalancer = lb
			break
		}
	}
	if loadbalancer.Id == "" {
		err := errors.New("no loadBalancer matches this ID")
		return nil, err
	}
	return loadbalancer, nil
}

func (hcnObj *mockHCN) createEndpoint(endpoint *hcn.HostComputeEndpoint, network *hcn.HostComputeNetwork) (*hcn.HostComputeEndpoint, error) {
	newEndpoint := &hcn.HostComputeEndpoint{
		Id:                   hcnObj.endpointIDs[hcnObj.epIDIndex],
		Name:                 hcnObj.endpointIDs[hcnObj.epIDIndex],
		HostComputeNetwork:   network.Id,
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
	hcnObj.epIDIndex++
	hcnObj.Endpoints = append(hcnObj.Endpoints, newEndpoint)

	return newEndpoint, nil
}

func (hcnObj *mockHCN) createRemoteEndpoint(endpoint *hcn.HostComputeEndpoint, network *hcn.HostComputeNetwork) (*hcn.HostComputeEndpoint, error) {
	newEndpoint := &hcn.HostComputeEndpoint{
		Id:                   hcnObj.endpointIDs[hcnObj.epIDIndex],
		Name:                 endpoint.Name,
		HostComputeNetwork:   hcnObj.hnsNetworkName,
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
	hcnObj.epIDIndex++
	hcnObj.Endpoints = append(hcnObj.Endpoints, newEndpoint)

	return newEndpoint, nil
}

func (hcnObj *mockHCN) createLoadBalancer(loadbalancer *hcn.HostComputeLoadBalancer) (*hcn.HostComputeLoadBalancer, error) {
	newLoadBalancer := &hcn.HostComputeLoadBalancer{
		Id:                   hcnObj.loadBalancerIDs[hcnObj.lbIndex],
		HostComputeEndpoints: loadbalancer.HostComputeEndpoints,
		SourceVIP:            loadbalancer.SourceVIP,
		Flags:                loadbalancer.Flags,
		FrontendVIPs:         loadbalancer.FrontendVIPs,
		PortMappings:         loadbalancer.PortMappings,
		SchemaVersion:        loadbalancer.SchemaVersion,
	}

	hcnObj.lbIndex++
	hcnObj.Loadbalancers = append(hcnObj.Loadbalancers, newLoadBalancer)

	return newLoadBalancer, nil
}

func (hcnObj *mockHCN) deleteLoadBalancer(loadbalancer *hcn.HostComputeLoadBalancer) error {
	var i int

	for _, lb := range hcnObj.Loadbalancers {
		i++
		if lb.Id == loadbalancer.Id {
			break
		}
	}

	i--

	if len(hcnObj.Loadbalancers) != 0 { //here we actually delete the load balancer from mockHCN memory
		copy(hcnObj.Loadbalancers[i:], hcnObj.Loadbalancers[i+1:])
		hcnObj.Loadbalancers[len(hcnObj.Loadbalancers)-1] = nil
		hcnObj.Loadbalancers = hcnObj.Loadbalancers[:len(hcnObj.Loadbalancers)-1]
	}

	return nil
}

func (hcnObj *mockHCN) deleteEndpoint(endpoint *hcn.HostComputeEndpoint) error {
	var i int

	for _, ep := range hcnObj.Endpoints {
		i++
		if ep.Id == endpoint.Id {
			break
		}
	}

	i--

	if len(hcnObj.Endpoints) != 0 { //here we actually delete the endpoint from mockHCN memory
		copy(hcnObj.Endpoints[i:], hcnObj.Endpoints[i+1:])
		hcnObj.Endpoints[len(hcnObj.Endpoints)-1] = nil
		hcnObj.Endpoints = hcnObj.Endpoints[:len(hcnObj.Endpoints)-1]
	}

	return nil
}
