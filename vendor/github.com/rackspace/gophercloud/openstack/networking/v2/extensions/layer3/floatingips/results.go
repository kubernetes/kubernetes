package floatingips

import (
	"fmt"

	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// FloatingIP represents a floating IP resource. A floating IP is an external
// IP address that is mapped to an internal port and, optionally, a specific
// IP address on a private network. In other words, it enables access to an
// instance on a private network from an external network. For this reason,
// floating IPs can only be defined on networks where the `router:external'
// attribute (provided by the external network extension) is set to True.
type FloatingIP struct {
	// Unique identifier for the floating IP instance.
	ID string `json:"id" mapstructure:"id"`

	// UUID of the external network where the floating IP is to be created.
	FloatingNetworkID string `json:"floating_network_id" mapstructure:"floating_network_id"`

	// Address of the floating IP on the external network.
	FloatingIP string `json:"floating_ip_address" mapstructure:"floating_ip_address"`

	// UUID of the port on an internal network that is associated with the floating IP.
	PortID string `json:"port_id" mapstructure:"port_id"`

	// The specific IP address of the internal port which should be associated
	// with the floating IP.
	FixedIP string `json:"fixed_ip_address" mapstructure:"fixed_ip_address"`

	// Owner of the floating IP. Only admin users can specify a tenant identifier
	// other than its own.
	TenantID string `json:"tenant_id" mapstructure:"tenant_id"`

	// The condition of the API resource.
	Status string `json:"status" mapstructure:"status"`
}

type commonResult struct {
	gophercloud.Result
}

// Extract a result and extracts a FloatingIP resource.
func (r commonResult) Extract() (*FloatingIP, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var res struct {
		FloatingIP *FloatingIP `json:"floatingip"`
	}

	err := mapstructure.Decode(r.Body, &res)
	if err != nil {
		return nil, fmt.Errorf("Error decoding Neutron floating IP: %v", err)
	}

	return res.FloatingIP, nil
}

// CreateResult represents the result of a create operation.
type CreateResult struct {
	commonResult
}

// GetResult represents the result of a get operation.
type GetResult struct {
	commonResult
}

// UpdateResult represents the result of an update operation.
type UpdateResult struct {
	commonResult
}

// DeleteResult represents the result of an update operation.
type DeleteResult struct {
	gophercloud.ErrResult
}

// FloatingIPPage is the page returned by a pager when traversing over a
// collection of floating IPs.
type FloatingIPPage struct {
	pagination.LinkedPageBase
}

// NextPageURL is invoked when a paginated collection of floating IPs has reached
// the end of a page and the pager seeks to traverse over a new one. In order
// to do this, it needs to construct the next page's URL.
func (p FloatingIPPage) NextPageURL() (string, error) {
	type resp struct {
		Links []gophercloud.Link `mapstructure:"floatingips_links"`
	}

	var r resp
	err := mapstructure.Decode(p.Body, &r)
	if err != nil {
		return "", err
	}

	return gophercloud.ExtractNextURL(r.Links)
}

// IsEmpty checks whether a NetworkPage struct is empty.
func (p FloatingIPPage) IsEmpty() (bool, error) {
	is, err := ExtractFloatingIPs(p)
	if err != nil {
		return true, nil
	}
	return len(is) == 0, nil
}

// ExtractFloatingIPs accepts a Page struct, specifically a FloatingIPPage struct,
// and extracts the elements into a slice of FloatingIP structs. In other words,
// a generic collection is mapped into a relevant slice.
func ExtractFloatingIPs(page pagination.Page) ([]FloatingIP, error) {
	var resp struct {
		FloatingIPs []FloatingIP `mapstructure:"floatingips" json:"floatingips"`
	}

	err := mapstructure.Decode(page.(FloatingIPPage).Body, &resp)

	return resp.FloatingIPs, err
}
