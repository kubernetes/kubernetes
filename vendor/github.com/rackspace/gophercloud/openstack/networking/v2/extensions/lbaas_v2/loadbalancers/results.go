package loadbalancers

import (
	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas_v2/listeners"
	"github.com/rackspace/gophercloud/pagination"
)

// LoadBalancer is the primary load balancing configuration object that specifies
// the virtual IP address on which client traffic is received, as well
// as other details such as the load balancing method to be use, protocol, etc.
type LoadBalancer struct {
	// Human-readable description for the Loadbalancer.
	Description string `mapstructure:"description" json:"description"`

	// The administrative state of the Loadbalancer. A valid value is true (UP) or false (DOWN).
	AdminStateUp bool `mapstructure:"admin_state_up" json:"admin_state_up"`

	// Owner of the LoadBalancer. Only an admin user can specify a tenant ID other than its own.
	TenantID string `mapstructure:"tenant_id" json:"tenant_id"`

	// The provisioning status of the LoadBalancer. This value is ACTIVE, PENDING_CREATE or ERROR.
	ProvisioningStatus string `mapstructure:"provisioning_status" json:"provisioning_status"`

	// The IP address of the Loadbalancer.
	VipAddress string `mapstructure:"vip_address" json:"vip_address"`

	// The UUID of the subnet on which to allocate the virtual IP for the Loadbalancer address.
	VipSubnetID string `mapstructure:"vip_subnet_id" json:"vip_subnet_id"`

	// The unique ID for the LoadBalancer.
	ID string `mapstructure:"id" json:"id"`

	// The operating status of the LoadBalancer. This value is ONLINE or OFFLINE.
	OperatingStatus string `mapstructure:"operating_status" json:"operating_status"`

	// Human-readable name for the LoadBalancer. Does not have to be unique.
	Name string `mapstructure:"name" json:"name"`

	// The UUID of a flavor if set.
	Flavor string `mapstructure:"flavor" json:"flavor"`

	// The name of the provider.
	Provider string `mapstructure:"provider" json:"provider"`

	Listeners []listeners.Listener `mapstructure:"listeners" json:"listeners"`
}

type StatusTree struct {
	Loadbalancer *LoadBalancer `mapstructure:"loadbalancer" json:"loadbalancer"`
}

// LoadbalancerPage is the page returned by a pager when traversing over a
// collection of routers.
type LoadbalancerPage struct {
	pagination.LinkedPageBase
}

// NextPageURL is invoked when a paginated collection of routers has reached
// the end of a page and the pager seeks to traverse over a new one. In order
// to do this, it needs to construct the next page's URL.
func (p LoadbalancerPage) NextPageURL() (string, error) {
	type resp struct {
		Links []gophercloud.Link `mapstructure:"loadbalancers_links"`
	}

	var r resp
	err := mapstructure.Decode(p.Body, &r)
	if err != nil {
		return "", err
	}

	return gophercloud.ExtractNextURL(r.Links)
}

// IsEmpty checks whether a RouterPage struct is empty.
func (p LoadbalancerPage) IsEmpty() (bool, error) {
	is, err := ExtractLoadbalancers(p)
	if err != nil {
		return true, nil
	}
	return len(is) == 0, nil
}

// ExtractLoadbalancers accepts a Page struct, specifically a LoadbalancerPage struct,
// and extracts the elements into a slice of LoadBalancer structs. In other words,
// a generic collection is mapped into a relevant slice.
func ExtractLoadbalancers(page pagination.Page) ([]LoadBalancer, error) {
	var resp struct {
		LoadBalancers []LoadBalancer `mapstructure:"loadbalancers" json:"loadbalancers"`
	}
	err := mapstructure.Decode(page.(LoadbalancerPage).Body, &resp)

	return resp.LoadBalancers, err
}

type commonResult struct {
	gophercloud.Result
}

// Extract is a function that accepts a result and extracts a router.
func (r commonResult) Extract() (*LoadBalancer, error) {
	if r.Err != nil {
		return nil, r.Err
	}
	var res struct {
		LoadBalancer *LoadBalancer `mapstructure:"loadbalancer" json:"loadbalancer"`
	}
	err := mapstructure.Decode(r.Body, &res)

	return res.LoadBalancer, err
}

// Extract is a function that accepts a result and extracts a Loadbalancer.
func (r commonResult) ExtractStatuses() (*StatusTree, error) {
	if r.Err != nil {
		return nil, r.Err
	}
	var res struct {
		LoadBalancer *StatusTree `mapstructure:"statuses" json:"statuses"`
	}
	err := mapstructure.Decode(r.Body, &res)

	return res.LoadBalancer, err
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
