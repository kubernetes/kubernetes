package subnets

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

type commonResult struct {
	gophercloud.Result
}

// Extract is a function that accepts a result and extracts a subnet resource.
func (r commonResult) Extract() (*Subnet, error) {
	var s struct {
		Subnet *Subnet `json:"subnet"`
	}
	err := r.ExtractInto(&s)
	return s.Subnet, err
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

// DeleteResult represents the result of a delete operation.
type DeleteResult struct {
	gophercloud.ErrResult
}

// AllocationPool represents a sub-range of cidr available for dynamic
// allocation to ports, e.g. {Start: "10.0.0.2", End: "10.0.0.254"}
type AllocationPool struct {
	Start string `json:"start"`
	End   string `json:"end"`
}

// HostRoute represents a route that should be used by devices with IPs from
// a subnet (not including local subnet route).
type HostRoute struct {
	DestinationCIDR string `json:"destination"`
	NextHop         string `json:"nexthop"`
}

// Subnet represents a subnet. See package documentation for a top-level
// description of what this is.
type Subnet struct {
	// UUID representing the subnet
	ID string `json:"id"`
	// UUID of the parent network
	NetworkID string `json:"network_id"`
	// Human-readable name for the subnet. Might not be unique.
	Name string `json:"name"`
	// IP version, either `4' or `6'
	IPVersion int `json:"ip_version"`
	// CIDR representing IP range for this subnet, based on IP version
	CIDR string `json:"cidr"`
	// Default gateway used by devices in this subnet
	GatewayIP string `json:"gateway_ip"`
	// DNS name servers used by hosts in this subnet.
	DNSNameservers []string `json:"dns_nameservers"`
	// Sub-ranges of CIDR available for dynamic allocation to ports. See AllocationPool.
	AllocationPools []AllocationPool `json:"allocation_pools"`
	// Routes that should be used by devices with IPs from this subnet (not including local subnet route).
	HostRoutes []HostRoute `json:"host_routes"`
	// Specifies whether DHCP is enabled for this subnet or not.
	EnableDHCP bool `json:"enable_dhcp"`
	// Owner of network. Only admin users can specify a tenant_id other than its own.
	TenantID string `json:"tenant_id"`
}

// SubnetPage is the page returned by a pager when traversing over a collection
// of subnets.
type SubnetPage struct {
	pagination.LinkedPageBase
}

// NextPageURL is invoked when a paginated collection of subnets has reached
// the end of a page and the pager seeks to traverse over a new one. In order
// to do this, it needs to construct the next page's URL.
func (r SubnetPage) NextPageURL() (string, error) {
	var s struct {
		Links []gophercloud.Link `json:"subnets_links"`
	}
	err := r.ExtractInto(&s)
	if err != nil {
		return "", err
	}
	return gophercloud.ExtractNextURL(s.Links)
}

// IsEmpty checks whether a SubnetPage struct is empty.
func (r SubnetPage) IsEmpty() (bool, error) {
	is, err := ExtractSubnets(r)
	return len(is) == 0, err
}

// ExtractSubnets accepts a Page struct, specifically a SubnetPage struct,
// and extracts the elements into a slice of Subnet structs. In other words,
// a generic collection is mapped into a relevant slice.
func ExtractSubnets(r pagination.Page) ([]Subnet, error) {
	var s struct {
		Subnets []Subnet `json:"subnets"`
	}
	err := (r.(SubnetPage)).ExtractInto(&s)
	return s.Subnets, err
}
