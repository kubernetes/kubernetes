package external

import (
	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud/openstack/networking/v2/networks"
	"github.com/rackspace/gophercloud/pagination"
)

// NetworkExternal represents a decorated form of a Network with based on the
// "external-net" extension.
type NetworkExternal struct {
	// UUID for the network
	ID string `mapstructure:"id" json:"id"`

	// Human-readable name for the network. Might not be unique.
	Name string `mapstructure:"name" json:"name"`

	// The administrative state of network. If false (down), the network does not forward packets.
	AdminStateUp bool `mapstructure:"admin_state_up" json:"admin_state_up"`

	// Indicates whether network is currently operational. Possible values include
	// `ACTIVE', `DOWN', `BUILD', or `ERROR'. Plug-ins might define additional values.
	Status string `mapstructure:"status" json:"status"`

	// Subnets associated with this network.
	Subnets []string `mapstructure:"subnets" json:"subnets"`

	// Owner of network. Only admin users can specify a tenant_id other than its own.
	TenantID string `mapstructure:"tenant_id" json:"tenant_id"`

	// Specifies whether the network resource can be accessed by any tenant or not.
	Shared bool `mapstructure:"shared" json:"shared"`

	// Specifies whether the network is an external network or not.
	External bool `mapstructure:"router:external" json:"router:external"`
}

func commonExtract(e error, response interface{}) (*NetworkExternal, error) {
	if e != nil {
		return nil, e
	}

	var res struct {
		Network *NetworkExternal `json:"network"`
	}

	err := mapstructure.Decode(response, &res)

	return res.Network, err
}

// ExtractGet decorates a GetResult struct returned from a networks.Get()
// function with extended attributes.
func ExtractGet(r networks.GetResult) (*NetworkExternal, error) {
	return commonExtract(r.Err, r.Body)
}

// ExtractCreate decorates a CreateResult struct returned from a networks.Create()
// function with extended attributes.
func ExtractCreate(r networks.CreateResult) (*NetworkExternal, error) {
	return commonExtract(r.Err, r.Body)
}

// ExtractUpdate decorates a UpdateResult struct returned from a
// networks.Update() function with extended attributes.
func ExtractUpdate(r networks.UpdateResult) (*NetworkExternal, error) {
	return commonExtract(r.Err, r.Body)
}

// ExtractList accepts a Page struct, specifically a NetworkPage struct, and
// extracts the elements into a slice of NetworkExternal structs. In other
// words, a generic collection is mapped into a relevant slice.
func ExtractList(page pagination.Page) ([]NetworkExternal, error) {
	var resp struct {
		Networks []NetworkExternal `mapstructure:"networks" json:"networks"`
	}

	err := mapstructure.Decode(page.(networks.NetworkPage).Body, &resp)

	return resp.Networks, err
}
