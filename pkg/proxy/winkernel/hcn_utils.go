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
	"github.com/Microsoft/hcsshim/hcn"
)

type hcnUtilService interface {
	getNetworkByName(networkName string) (*hcn.HostComputeNetwork, error)
	createNetwork(network *hcn.HostComputeNetwork) (*hcn.HostComputeNetwork, error)
	deleteNetwork(network *hcn.HostComputeNetwork) error
	listEndpointsOfNetwork(networkId string) ([]hcn.HostComputeEndpoint, error)
	getEndpointByID(endpointId string) (*hcn.HostComputeEndpoint, error)
	listEndpoints() ([]hcn.HostComputeEndpoint, error)
	getEndpointByName(endpointName string) (*hcn.HostComputeEndpoint, error)
	listLoadBalancers() ([]hcn.HostComputeLoadBalancer, error)
	getLoadBalancerByID(loadBalancerId string) (*hcn.HostComputeLoadBalancer, error)
	createEndpoint(endpoint *hcn.HostComputeEndpoint, network *hcn.HostComputeNetwork) (*hcn.HostComputeEndpoint, error)
	createLoadBalancer(loadbalancer *hcn.HostComputeLoadBalancer) (*hcn.HostComputeLoadBalancer, error)
	createRemoteEndpoint(endpoint *hcn.HostComputeEndpoint, network *hcn.HostComputeNetwork) (*hcn.HostComputeEndpoint, error)
	deleteLoadBalancer(loadbalancer *hcn.HostComputeLoadBalancer) error
	deleteEndpoint(endpoint *hcn.HostComputeEndpoint) error
}

type hcnUtils struct {
}

func (hcn hcnUtils) createLoadbalancer(lb *hcn.HostComputeLoadBalancer) (*hcn.HostComputeLoadBalancer, error) {
	return lb.Create()
}

func (h *hcnUtils) getNetworkByName(networkName string) (*hcn.HostComputeNetwork, error) {
	return hcn.GetNetworkByName(networkName)
}

func (h *hcnUtils) createNetwork(network *hcn.HostComputeNetwork) (*hcn.HostComputeNetwork, error) {
	return network.Create()
}

func (h *hcnUtils) deleteNetwork(network *hcn.HostComputeNetwork) error {
	return network.Delete()
}

func (h *hcnUtils) listEndpointsOfNetwork(networkId string) ([]hcn.HostComputeEndpoint, error) {
	return hcn.ListEndpointsOfNetwork(networkId)
}

func (h *hcnUtils) getEndpointByID(endpointId string) (*hcn.HostComputeEndpoint, error) {
	return hcn.GetEndpointByID(endpointId)
}

func (h *hcnUtils) listEndpoints() ([]hcn.HostComputeEndpoint, error) {
	return hcn.ListEndpoints()
}

func (h *hcnUtils) getEndpointByName(endpointName string) (*hcn.HostComputeEndpoint, error) {
	return hcn.GetEndpointByName(endpointName)
}

func (h *hcnUtils) listLoadBalancers() ([]hcn.HostComputeLoadBalancer, error) {
	return hcn.ListLoadBalancers()
}

func (h *hcnUtils) getLoadBalancerByID(loadBalancerId string) (*hcn.HostComputeLoadBalancer, error) {
	return hcn.GetLoadBalancerByID(loadBalancerId)
}

func (h *hcnUtils) createEndpoint(endpoint *hcn.HostComputeEndpoint, network *hcn.HostComputeNetwork) (*hcn.HostComputeEndpoint, error) {
	return network.CreateEndpoint(endpoint)
}

func (h *hcnUtils) createRemoteEndpoint(endpoint *hcn.HostComputeEndpoint, network *hcn.HostComputeNetwork) (*hcn.HostComputeEndpoint, error) {
	return network.CreateRemoteEndpoint(endpoint)
}

func (h *hcnUtils) createLoadBalancer(loadbalancer *hcn.HostComputeLoadBalancer) (*hcn.HostComputeLoadBalancer, error) {
	return loadbalancer.Create()
}

func (h *hcnUtils) deleteLoadBalancer(loadbalancer *hcn.HostComputeLoadBalancer) error {
	return loadbalancer.Delete()
}

func (h *hcnUtils) deleteEndpoint(endpoint *hcn.HostComputeEndpoint) error {
	return endpoint.Delete()
}
