package external

import (
	"github.com/gophercloud/gophercloud/openstack/networking/v2/networks"
	"github.com/gophercloud/gophercloud/pagination"
)

// NetworkExternal represents a decorated form of a Network with based on the
// "external-net" extension.
type NetworkExternal struct {
	// UUID for the network
	ID string `json:"id"`

	// Human-readable name for the network. Might not be unique.
	Name string `json:"name"`

	// The administrative state of network. If false (down), the network does not forward packets.
	AdminStateUp bool `json:"admin_state_up"`

	// Indicates whether network is currently operational. Possible values include
	// `ACTIVE', `DOWN', `BUILD', or `ERROR'. Plug-ins might define additional values.
	Status string `json:"status"`

	// Subnets associated with this network.
	Subnets []string `json:"subnets"`

	// Owner of network. Only admin users can specify a tenant_id other than its own.
	TenantID string `json:"tenant_id"`

	// Specifies whether the network resource can be accessed by any tenant or not.
	Shared bool `json:"shared"`

	// Specifies whether the network is an external network or not.
	External bool `json:"router:external"`
}

// ExtractGet decorates a GetResult struct returned from a networks.Get()
// function with extended attributes.
func ExtractGet(r networks.GetResult) (*NetworkExternal, error) {
	var s struct {
		Network *NetworkExternal `json:"network"`
	}
	err := r.ExtractInto(&s)
	return s.Network, err
}

// ExtractCreate decorates a CreateResult struct returned from a networks.Create()
// function with extended attributes.
func ExtractCreate(r networks.CreateResult) (*NetworkExternal, error) {
	var s struct {
		Network *NetworkExternal `json:"network"`
	}
	err := r.ExtractInto(&s)
	return s.Network, err
}

// ExtractUpdate decorates a UpdateResult struct returned from a
// networks.Update() function with extended attributes.
func ExtractUpdate(r networks.UpdateResult) (*NetworkExternal, error) {
	var s struct {
		Network *NetworkExternal `json:"network"`
	}
	err := r.ExtractInto(&s)
	return s.Network, err
}

// ExtractList accepts a Page struct, specifically a NetworkPage struct, and
// extracts the elements into a slice of NetworkExternal structs. In other
// words, a generic collection is mapped into a relevant slice.
func ExtractList(r pagination.Page) ([]NetworkExternal, error) {
	var s struct {
		Networks []NetworkExternal `json:"networks" json:"networks"`
	}
	err := (r.(networks.NetworkPage)).ExtractInto(&s)
	return s.Networks, err
}
