//go:build windows

package hnslib

import (
	"github.com/Microsoft/hnslib/internal/hns"
)

// Subnet is associated with a network and represents a list
// of subnets available to the network
type Subnet = hns.Subnet

// MacPool is associated with a network and represents a list
// of macaddresses available to the network
type MacPool = hns.MacPool

// HNSNetwork represents a network in HNS
type HNSNetwork = hns.HNSNetwork

// HNSEndpoint represents a network endpoint in HNS
type HNSEndpoint = hns.HNSEndpoint

// HNSEndpointStats represent the stats for an networkendpoint in HNS
type HNSEndpointStats = hns.EndpointStats

// PolicyList is a structure defining schema for Policy list request
type PolicyList = hns.PolicyList

// Namespace represents a Compartment.
type Namespace = hns.Namespace

// HNSListNetworkRequest makes a HNS call to query the list of available networks
func HNSListNetworkRequest(method, path, request string) ([]HNSNetwork, error) {
	return hns.HNSListNetworkRequest(method, path, request)
}

// GetHNSEndpointStats gets the endpoint stats by ID
func GetHNSEndpointStats(endpointName string) (*HNSEndpointStats, error) {
	return hns.GetHNSEndpointStats(endpointName)
}

// HNSListEndpointRequest makes a HNS call to query the list of available endpoints
func HNSListEndpointRequest() ([]HNSEndpoint, error) {
	return hns.HNSListEndpointRequest()
}

// HNSListPolicyListRequest gets all the policy list
func HNSListPolicyListRequest() ([]PolicyList, error) {
	return hns.HNSListPolicyListRequest()
}