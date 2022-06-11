package hcntesting

import (
	"encoding/json"

	"github.com/Microsoft/hcsshim/hcn"
	"k8s.io/klog/v2"
)

const (
	guid = "123ABC"
)

//the fakeHCN saves the created endpoints and loadbalancers in slices to be able to work with them easier
type FakeHCN struct {
	Endpoints     []*hcn.HostComputeEndpoint
	Loadbalancers []*hcn.HostComputeLoadBalancer
}

func (HCN *FakeHCN) GetNetworkByName(networkName string) (*hcn.HostComputeNetwork, error) {
	policysettings := &hcn.RemoteSubnetRoutePolicySetting{
		DestinationPrefix:           "192.168.2.0/24",
		IsolationId:                 4096,
		ProviderAddress:             "10.0.0.3",
		DistributedRouterMacAddress: "00-11-22-33-44-55",
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
						IpAddressPrefix: "192.168.1.0/24",
						Routes: []hcn.Route{
							{
								NextHop:           "192.168.1.1",
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
			endpoint.Id = endpointId
			endpoint.Name = ep.Name
			endpoint.HostComputeNetwork = ep.HostComputeNetwork
			endpoint.HostComputeNamespace = ep.HostComputeNamespace
			endpoint.IpConfigurations = ep.IpConfigurations
			endpoint.Policies = ep.Policies
			endpoint.Dns = ep.Dns
			endpoint.Routes = ep.Routes
			endpoint.MacAddress = ep.MacAddress
			endpoint.Flags = ep.Flags
			endpoint.Health = ep.Health
			endpoint.SchemaVersion = ep.SchemaVersion
		}
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
			endpoint.Id = ep.Id
			endpoint.Name = endpointName
			endpoint.HostComputeNetwork = ep.HostComputeNetwork
			endpoint.Health = ep.Health
			endpoint.IpConfigurations = ep.IpConfigurations
			endpoint.HostComputeNamespace = ep.HostComputeNamespace
			endpoint.Policies = ep.Policies
			endpoint.Dns = ep.Dns
			endpoint.Routes = ep.Routes
			endpoint.MacAddress = ep.MacAddress
			endpoint.Flags = ep.Flags
			endpoint.SchemaVersion = ep.SchemaVersion
		}
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
			loadbalancer.Id = loadBalancerId
			loadbalancer.Flags = lb.Flags
			loadbalancer.HostComputeEndpoints = lb.HostComputeEndpoints
			loadbalancer.SourceVIP = lb.SourceVIP
			loadbalancer.SchemaVersion = lb.SchemaVersion
			loadbalancer.PortMappings = lb.PortMappings
			loadbalancer.FrontendVIPs = lb.FrontendVIPs
		}
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

	if len(HCN.Loadbalancers) != 0 {
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

	if len(HCN.Endpoints) != 0 {
		copy(HCN.Endpoints[i:], HCN.Endpoints[i+1:])
		HCN.Endpoints[len(HCN.Endpoints)-1] = nil
		HCN.Endpoints = HCN.Endpoints[:len(HCN.Endpoints)-1]
	}

	return nil
}
