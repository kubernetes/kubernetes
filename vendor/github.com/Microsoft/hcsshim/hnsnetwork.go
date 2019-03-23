package hcsshim

import (
	"github.com/Microsoft/hcsshim/internal/hns"
)

// Subnet is assoicated with a network and represents a list
// of subnets available to the network
type Subnet = hns.Subnet

// MacPool is assoicated with a network and represents a list
// of macaddresses available to the network
type MacPool = hns.MacPool

// HNSNetwork represents a network in HNS
type HNSNetwork = hns.HNSNetwork

// HNSNetworkRequest makes a call into HNS to update/query a single network
func HNSNetworkRequest(method, path, request string) (*HNSNetwork, error) {
	return hns.HNSNetworkRequest(method, path, request)
}

// HNSListNetworkRequest makes a HNS call to query the list of available networks
func HNSListNetworkRequest(method, path, request string) ([]HNSNetwork, error) {
	return hns.HNSListNetworkRequest(method, path, request)
}

// GetHNSNetworkByID
func GetHNSNetworkByID(networkID string) (*HNSNetwork, error) {
	return hns.GetHNSNetworkByID(networkID)
}

// GetHNSNetworkName filtered by Name
func GetHNSNetworkByName(networkName string) (*HNSNetwork, error) {
	return hns.GetHNSNetworkByName(networkName)
}
