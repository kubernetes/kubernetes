package pools

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// Pool represents a logical set of devices, such as web servers, that you
// group together to receive and process traffic. The load balancing function
// chooses a member of the pool according to the configured load balancing
// method to handle the new requests or connections received on the VIP address.
// There is only one pool per virtual IP.
type Pool struct {
	// Status of the pool. Indicates whether the pool is operational.
	Status string

	// LBMethod is the load-balancer algorithm, which is round-robin,
	// least-connections, and so on. This value, which must be supported, is
	// dependent on the provider.
	LBMethod string `json:"lb_method"`

	// Protocol of the pool, which is TCP, HTTP, or HTTPS.
	Protocol string

	// Description for the pool.
	Description string

	// MonitorIDs are the IDs of associated monitors which check the health of
	// the pool members.
	MonitorIDs []string `json:"health_monitors"`

	// SubnetID is the network on which the members of the pool will be located.
	// Only members that are on this network can be added to the pool.
	SubnetID string `json:"subnet_id"`

	// TenantID is the owner of the pool.
	TenantID string `json:"tenant_id"`

	// AdminStateUp is the administrative state of the pool, which is up
	// (true) or down (false).
	AdminStateUp bool `json:"admin_state_up"`

	// Name of the pool.
	Name string

	// MemberIDs is the list of member IDs that belong to the pool.
	MemberIDs []string `json:"members"`

	// ID is the unique ID for the pool.
	ID string

	// VIPID is the ID of the virtual IP associated with this pool.
	VIPID string `json:"vip_id"`

	// The provider.
	Provider string
}

// PoolPage is the page returned by a pager when traversing over a
// collection of pools.
type PoolPage struct {
	pagination.LinkedPageBase
}

// NextPageURL is invoked when a paginated collection of pools has reached
// the end of a page and the pager seeks to traverse over a new one. In order
// to do this, it needs to construct the next page's URL.
func (r PoolPage) NextPageURL() (string, error) {
	var s struct {
		Links []gophercloud.Link `json:"pools_links"`
	}
	err := r.ExtractInto(&s)
	if err != nil {
		return "", err
	}
	return gophercloud.ExtractNextURL(s.Links)
}

// IsEmpty checks whether a PoolPage struct is empty.
func (r PoolPage) IsEmpty() (bool, error) {
	is, err := ExtractPools(r)
	return len(is) == 0, err
}

// ExtractPools accepts a Page struct, specifically a PoolPage struct,
// and extracts the elements into a slice of Router structs. In other words,
// a generic collection is mapped into a relevant slice.
func ExtractPools(r pagination.Page) ([]Pool, error) {
	var s struct {
		Pools []Pool `json:"pools"`
	}
	err := (r.(PoolPage)).ExtractInto(&s)
	return s.Pools, err
}

type commonResult struct {
	gophercloud.Result
}

// Extract is a function that accepts a result and extracts a router.
func (r commonResult) Extract() (*Pool, error) {
	var s struct {
		Pool *Pool `json:"pool"`
	}
	err := r.ExtractInto(&s)
	return s.Pool, err
}

// CreateResult represents the result of a create operation. Call its Extract
// method to interpret it as a Pool.
type CreateResult struct {
	commonResult
}

// GetResult represents the result of a get operation. Call its Extract
// method to interpret it as a Pool.
type GetResult struct {
	commonResult
}

// UpdateResult represents the result of an update operation. Call its Extract
// method to interpret it as a Pool.
type UpdateResult struct {
	commonResult
}

// DeleteResult represents the result of a delete operation. Call its
// ExtractErr method to interpret it as a Pool.
type DeleteResult struct {
	gophercloud.ErrResult
}

// AssociateResult represents the result of an association operation. Call its Extract
// method to interpret it as a Pool.
type AssociateResult struct {
	commonResult
}
