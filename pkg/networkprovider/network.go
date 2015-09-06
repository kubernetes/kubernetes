/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package networkprovider

import (
	"errors"

	"k8s.io/kubernetes/pkg/api"
)

const (
	namePrefix = "kube"
)

var ErrNotFound = errors.New("NotFound")
var ErrMultipleResults = errors.New("MultipleResults")

// Interface is an abstract, pluggable interface for network providers.
type Interface interface {
	// Pods returns a pod interface
	Pods() Pods
	// Networks returns a network interface
	Networks() Networks
	// LoadBalancer returns a balancer interface
	LoadBalancers() LoadBalancers
	// ProviderName returns the network provider ID.
	ProviderName() string
	// CheckTenantID
	CheckTenantID(tenantID string) (bool, error)
}

// Network is a representation of a network segment
type Network struct {
	// Name is the name of the network
	Name string `json:"name"`
	// UID is the id of the network
	UID string `json:"id,omitempty"`
	// SegmentID is the segment id of the network
	SegmentID int `json:"segmentID,omitempty"`
	// Subnets is a list of Subnet belongs to the network
	Subnets []*Subnet `json:"subnets,omitempty"`
	// TenantID is the tenant id of the network
	TenantID string `json:"tenantID,omitempty"`
	// Status is the status fo the network
	Status NetworkStatus `json:"status"`
}

// Subnet is a representaion of a subnet
type Subnet struct {
	// Name is the name of the subnet
	Name string `json:"name"`
	// UID is the id of the subnet
	UID string `json:"id,omitempty"`
	// CIDR is the subnet cidr of the subnet
	CIDR string `json:"cidr"`
	// Gateway is the gateway of the subnet
	Gateway string `json:"gateway,omitempty"`
	// Routes is a list of routes of the subnet
	Routes []*Route `json:"routes,omitempty"`
	// DNSNameServer is a list of dns nameservers of the subnet
	DNSNameServer []string `json:"dnsNameservers,omitempty"`
	// TenantID is the tenant id of the subnet
	TenantID string `json:"tenantID,omitempty"`
}

// Route is a representation of an advanced routing rule.
type Route struct {
	// Name is the name of the routing rule in the subnet
	Name string `json:"name,omitempty"`
	// TNexthop is the nexthop the specified route
	Nexthop string `json:"nexthop"`
	// Destination CIDR is the CIDR format IP range that this routing rule
	// applies to.
	DestinationCIDR string `json:"destinationCIDR,omitempty"`
}

type Pods interface {
	// Setup pod
	SetupPod(podName, namespace, podInfraContainerID string, network *Network, containerRuntime string) error
	// Teardown pod
	TeardownPod(podName, namespace, podInfraContainerID string, network *Network, containerRuntime string) error
	// Status of pod
	PodStatus(podName, namespace, podInfraContainerID string, network *Network, containerRuntime string) (string, error)
}

// Networks is an abstract, pluggable interface for network segment
type Networks interface {
	// Get network by networkName
	GetNetwork(networkName string) (*Network, error)
	// Get network by networkID
	GetNetworkByID(networkID string) (*Network, error)
	// Create network
	CreateNetwork(network *Network) error
	// Update network
	UpdateNetwork(network *Network) error
	// Delete network by networkName
	DeleteNetwork(networkName string) error
}

// LoadBalancerType is the type of the load balancer
type LoadBalancerType string

const (
	LoadBalancerTypeTCP   LoadBalancerType = "TCP"
	LoadBalancerTypeUDP   LoadBalancerType = "UDP"
	LoadBalancerTypeHTTP  LoadBalancerType = "HTTP"
	LoadBalancerTypeHTTPS LoadBalancerType = "HTTPS"
)

type NetworkStatus string

const (
	// NetworkInitializing means the network is just accepted by system
	NetworkInitializing NetworkStatus = "Initializing"
	// NetworkActive means the network is available for use in the system
	NetworkActive NetworkStatus = "Active"
	// NetworkPending means the network is accepted by system, but it is still
	// processing by network provider
	NetworkPending NetworkStatus = "Pending"
	// NetworkFailed means the network is not available
	NetworkFailed NetworkStatus = "Failed"
	// NetworkTerminating means the network is undergoing graceful termination
	NetworkTerminating NetworkStatus = "Terminating"
)

type HostPort struct {
	Name        string `json:"name,omitempty"`
	ServicePort int    `json:"servicePort"`
	IPAddress   string `json:"ipAddress"`
	TargetPort  int    `json:"port"`
}

// LoadBalancer is a replace of kube-proxy, so load-balancing can be handled
// by network providers so as to overcome iptables overhead
type LoadBalancer struct {
	// the name of the load balancer
	Name string `json:"name"`
	// the id of the load balancer
	UID string `json:"id,omitempty"`
	// the type of the load balancer
	Type LoadBalancerType `json:"loadBalanceType,omitempty"`
	// the vip of the load balancer
	Vip string `json:"vip,omitempty"`
	// the external ip of the load balancer
	ExternalIPs []string `json:"externalIPs,omitempty"`
	// subnet of the load balancer
	Subnets []*Subnet `json:"subnets"`
	// hosts and ports of the load balancer
	Hosts []*HostPort `json:"hosts"`
	// status
	Status string `json:"status"`
	// TenantID is the tenant id of the network
	TenantID string `json:"tenantID"`
}

// LoadBalancers is an abstract, pluggable interface for load balancers.
type LoadBalancers interface {
	// Get load balancer by name
	GetLoadBalancer(name string) (*LoadBalancer, error)
	// Create load balancer, return vip and externalIP
	CreateLoadBalancer(loadBalancer *LoadBalancer, affinity api.ServiceAffinity) (string, error)
	// Update load balancer, return vip and externalIP
	UpdateLoadBalancer(name string, hosts []*HostPort, externalIPs []string) (string, error)
	// Delete load balancer
	DeleteLoadBalancer(name string) error
}

func BuildNetworkName(name, tenantID string) string {
	return namePrefix + "_" + name + "_" + tenantID
}

func BuildLoadBalancerName(name, namespace string) string {
	return namePrefix + "_" + name + "_" + namespace
}

func ApiNetworkToProviderNetwork(net *api.Network) *Network {
	pdNetwork := Network{
		Name:     BuildNetworkName(net.Name, net.Spec.TenantID),
		TenantID: net.Spec.TenantID,
		Subnets:  make([]*Subnet, 0, 1),
	}

	for key, subnet := range net.Spec.Subnets {
		s := Subnet{
			CIDR:     subnet.CIDR,
			Gateway:  subnet.Gateway,
			Name:     BuildNetworkName(key, net.Spec.TenantID),
			TenantID: net.Spec.TenantID,
		}
		pdNetwork.Subnets = append(pdNetwork.Subnets, &s)
	}

	return &pdNetwork
}
