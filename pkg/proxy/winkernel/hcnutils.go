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
	"github.com/Microsoft/hnslib/hcn"

	"k8s.io/klog/v2"
)

type HcnService interface {
	// Network functions
	GetNetworkByName(networkName string) (*hcn.HostComputeNetwork, error)
	GetNetworkByID(networkID string) (*hcn.HostComputeNetwork, error)
	// Endpoint functions
	ListEndpoints() ([]hcn.HostComputeEndpoint, error)
	ListEndpointsOfNetwork(networkId string) ([]hcn.HostComputeEndpoint, error)
	GetEndpointByID(endpointId string) (*hcn.HostComputeEndpoint, error)
	GetEndpointByName(endpointName string) (*hcn.HostComputeEndpoint, error)
	CreateEndpoint(network *hcn.HostComputeNetwork, endpoint *hcn.HostComputeEndpoint) (*hcn.HostComputeEndpoint, error)
	CreateRemoteEndpoint(network *hcn.HostComputeNetwork, endpoint *hcn.HostComputeEndpoint) (*hcn.HostComputeEndpoint, error)
	DeleteEndpoint(endpoint *hcn.HostComputeEndpoint) error
	// LoadBalancer functions
	ListLoadBalancers() ([]hcn.HostComputeLoadBalancer, error)
	GetLoadBalancerByID(loadBalancerId string) (*hcn.HostComputeLoadBalancer, error)
	CreateLoadBalancer(loadBalancer *hcn.HostComputeLoadBalancer) (*hcn.HostComputeLoadBalancer, error)
	UpdateLoadBalancer(loadBalancer *hcn.HostComputeLoadBalancer, hnsID string) (*hcn.HostComputeLoadBalancer, error)
	DeleteLoadBalancer(loadBalancer *hcn.HostComputeLoadBalancer) error
	// Features functions
	GetSupportedFeatures() hcn.SupportedFeatures
	Ipv6DualStackSupported() error
	DsrSupported() error
	// Policy functions
	DeleteAllHnsLoadBalancerPolicy()
	RemoteSubnetSupported() error
}

type hcnImpl struct{}

func newHcnImpl() hcnImpl {
	return hcnImpl{}
}

func (hcnObj hcnImpl) GetNetworkByName(networkName string) (*hcn.HostComputeNetwork, error) {
	return hcn.GetNetworkByName(networkName)
}

func (hcnObj hcnImpl) GetNetworkByID(networkID string) (*hcn.HostComputeNetwork, error) {
	return hcn.GetNetworkByID(networkID)
}

func (hcnObj hcnImpl) ListEndpoints() ([]hcn.HostComputeEndpoint, error) {
	return hcn.ListEndpoints()
}

func (hcnObj hcnImpl) ListEndpointsOfNetwork(networkId string) ([]hcn.HostComputeEndpoint, error) {
	return hcn.ListEndpointsOfNetwork(networkId)
}

func (hcnObj hcnImpl) GetEndpointByID(endpointId string) (*hcn.HostComputeEndpoint, error) {
	return hcn.GetEndpointByID(endpointId)
}

func (hcnObj hcnImpl) GetEndpointByName(endpointName string) (*hcn.HostComputeEndpoint, error) {
	return hcn.GetEndpointByName(endpointName)
}

func (hcnObj hcnImpl) CreateEndpoint(network *hcn.HostComputeNetwork, endpoint *hcn.HostComputeEndpoint) (*hcn.HostComputeEndpoint, error) {
	return network.CreateEndpoint(endpoint)
}

func (hcnObj hcnImpl) CreateRemoteEndpoint(network *hcn.HostComputeNetwork, endpoint *hcn.HostComputeEndpoint) (*hcn.HostComputeEndpoint, error) {
	return network.CreateRemoteEndpoint(endpoint)
}

func (hcnObj hcnImpl) DeleteEndpoint(endpoint *hcn.HostComputeEndpoint) error {
	return endpoint.Delete()
}

func (hcnObj hcnImpl) ListLoadBalancers() ([]hcn.HostComputeLoadBalancer, error) {
	return hcn.ListLoadBalancers()
}

func (hcnObj hcnImpl) GetLoadBalancerByID(loadBalancerId string) (*hcn.HostComputeLoadBalancer, error) {
	return hcn.GetLoadBalancerByID(loadBalancerId)
}

func (hcnObj hcnImpl) CreateLoadBalancer(loadBalancer *hcn.HostComputeLoadBalancer) (*hcn.HostComputeLoadBalancer, error) {
	return loadBalancer.Create()
}

func (hcnObj hcnImpl) UpdateLoadBalancer(loadBalancer *hcn.HostComputeLoadBalancer, hnsID string) (*hcn.HostComputeLoadBalancer, error) {
	return loadBalancer.Update(hnsID)
}

func (hcnObj hcnImpl) DeleteLoadBalancer(loadBalancer *hcn.HostComputeLoadBalancer) error {
	return loadBalancer.Delete()
}

func (hcnObj hcnImpl) GetSupportedFeatures() hcn.SupportedFeatures {
	return hcn.GetSupportedFeatures()
}

func (hcnObj hcnImpl) Ipv6DualStackSupported() error {
	return hcn.IPv6DualStackSupported()
}

func (hcnObj hcnImpl) DsrSupported() error {
	return hcn.DSRSupported()
}

func (hcnObj hcnImpl) DeleteAllHnsLoadBalancerPolicy() {
	lbs, err := hcnObj.ListLoadBalancers()
	if err != nil {
		klog.V(2).ErrorS(err, "Deleting all existing loadbalancers failed.")
		return
	}
	klog.V(3).InfoS("Deleting all existing loadbalancers", "lbCount", len(lbs))
	for _, lb := range lbs {
		err = lb.Delete()
		if err != nil {
			klog.V(2).ErrorS(err, "Error deleting existing loadbalancer", "lb", lb)
		}
	}
}

func (hcnObj hcnImpl) RemoteSubnetSupported() error {
	return hcn.RemoteSubnetSupported()
}
