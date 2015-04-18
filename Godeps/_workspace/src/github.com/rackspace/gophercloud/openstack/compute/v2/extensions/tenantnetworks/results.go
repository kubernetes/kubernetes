package tenantnetworks

import (
	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// A Network represents a nova-network that an instance communicates on
type Network struct {
	// CIDR is the IPv4 subnet.
	CIDR string `mapstructure:"cidr"`

	// ID is the UUID of the network.
	ID string `mapstructure:"id"`

	// Name is the common name that the network has.
	Name string `mapstructure:"label"`
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
func ExtractNetworks(page pagination.Page) ([]Network, error) {
	networks := page.(NetworkPage).Body
	var res struct {
		Networks []Network `mapstructure:"networks"`
	}

	err := mapstructure.WeakDecode(networks, &res)

	return res.Networks, err
}

type NetworkResult struct {
	gophercloud.Result
}

// Extract is a method that attempts to interpret any Network resource
// response as a Network struct.
func (r NetworkResult) Extract() (*Network, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var res struct {
		Network *Network `json:"network" mapstructure:"network"`
	}

	err := mapstructure.Decode(r.Body, &res)
	return res.Network, err
}

// GetResult is the response from a Get operation. Call its Extract method to interpret it
// as a Network.
type GetResult struct {
	NetworkResult
}
