package loadbalancers

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/lbaas_v2/listeners"
	"github.com/gophercloud/gophercloud/pagination"
)

// LoadBalancer is the primary load balancing configuration object that specifies
// the virtual IP address on which client traffic is received, as well
// as other details such as the load balancing method to be use, protocol, etc.
type LoadBalancer struct {
	// Human-readable description for the Loadbalancer.
	Description string `json:"description"`
	// The administrative state of the Loadbalancer. A valid value is true (UP) or false (DOWN).
	AdminStateUp bool `json:"admin_state_up"`
	// Owner of the LoadBalancer. Only an admin user can specify a tenant ID other than its own.
	TenantID string `json:"tenant_id"`
	// The provisioning status of the LoadBalancer. This value is ACTIVE, PENDING_CREATE or ERROR.
	ProvisioningStatus string `json:"provisioning_status"`
	// The IP address of the Loadbalancer.
	VipAddress string `json:"vip_address"`
	// The UUID of the port associated with the IP address.
	VipPortID string `json:"vip_port_id"`
	// The UUID of the subnet on which to allocate the virtual IP for the Loadbalancer address.
	VipSubnetID string `json:"vip_subnet_id"`
	// The unique ID for the LoadBalancer.
	ID string `json:"id"`
	// The operating status of the LoadBalancer. This value is ONLINE or OFFLINE.
	OperatingStatus string `json:"operating_status"`
	// Human-readable name for the LoadBalancer. Does not have to be unique.
	Name string `json:"name"`
	// The UUID of a flavor if set.
	Flavor string `json:"flavor"`
	// The name of the provider.
	Provider  string               `json:"provider"`
	Listeners []listeners.Listener `json:"listeners"`
}

type StatusTree struct {
	Loadbalancer *LoadBalancer `json:"loadbalancer"`
}

// LoadBalancerPage is the page returned by a pager when traversing over a
// collection of routers.
type LoadBalancerPage struct {
	pagination.LinkedPageBase
}

// NextPageURL is invoked when a paginated collection of routers has reached
// the end of a page and the pager seeks to traverse over a new one. In order
// to do this, it needs to construct the next page's URL.
func (r LoadBalancerPage) NextPageURL() (string, error) {
	var s struct {
		Links []gophercloud.Link `json:"loadbalancers_links"`
	}
	err := r.ExtractInto(&s)
	if err != nil {
		return "", err
	}
	return gophercloud.ExtractNextURL(s.Links)
}

// IsEmpty checks whether a LoadBalancerPage struct is empty.
func (p LoadBalancerPage) IsEmpty() (bool, error) {
	is, err := ExtractLoadBalancers(p)
	return len(is) == 0, err
}

// ExtractLoadBalancers accepts a Page struct, specifically a LoadbalancerPage struct,
// and extracts the elements into a slice of LoadBalancer structs. In other words,
// a generic collection is mapped into a relevant slice.
func ExtractLoadBalancers(r pagination.Page) ([]LoadBalancer, error) {
	var s struct {
		LoadBalancers []LoadBalancer `json:"loadbalancers"`
	}
	err := (r.(LoadBalancerPage)).ExtractInto(&s)
	return s.LoadBalancers, err
}

type commonResult struct {
	gophercloud.Result
}

// Extract is a function that accepts a result and extracts a router.
func (r commonResult) Extract() (*LoadBalancer, error) {
	var s struct {
		LoadBalancer *LoadBalancer `json:"loadbalancer"`
	}
	err := r.ExtractInto(&s)
	return s.LoadBalancer, err
}

type GetStatusesResult struct {
	gophercloud.Result
}

// Extract is a function that accepts a result and extracts a Loadbalancer.
func (r GetStatusesResult) Extract() (*StatusTree, error) {
	var s struct {
		Statuses *StatusTree `json:"statuses"`
	}
	err := r.ExtractInto(&s)
	return s.Statuses, err
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
