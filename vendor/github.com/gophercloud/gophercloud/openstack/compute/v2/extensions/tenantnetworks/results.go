package tenantnetworks

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// A Network represents a nova-network that an instance communicates on
type Network struct {
	// CIDR is the IPv4 subnet.
	CIDR string `json:"cidr"`

	// ID is the UUID of the network.
	ID string `json:"id"`

	// Name is the common name that the network has.
	Name string `json:"label"`
}

// NetworkPage stores a single, only page of Networks
// results from a List call.
type NetworkPage struct {
	pagination.SinglePageBase
}

// IsEmpty determines whether or not a NetworkPage is empty.
func (page NetworkPage) IsEmpty() (bool, error) {
	va, err := ExtractNetworks(page)
	return len(va) == 0, err
}

// ExtractNetworks interprets a page of results as a slice of Networks
func ExtractNetworks(r pagination.Page) ([]Network, error) {
	var s struct {
		Networks []Network `json:"networks"`
	}
	err := (r.(NetworkPage)).ExtractInto(&s)
	return s.Networks, err
}

type NetworkResult struct {
	gophercloud.Result
}

// Extract is a method that attempts to interpret any Network resource
// response as a Network struct.
func (r NetworkResult) Extract() (*Network, error) {
	var s struct {
		Network *Network `json:"network"`
	}
	err := r.ExtractInto(&s)
	return s.Network, err
}

// GetResult is the response from a Get operation. Call its Extract method to interpret it
// as a Network.
type GetResult struct {
	NetworkResult
}
