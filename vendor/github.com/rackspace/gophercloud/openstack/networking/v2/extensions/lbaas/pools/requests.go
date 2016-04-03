package pools

import (
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// ListOpts allows the filtering and sorting of paginated collections through
// the API. Filtering is achieved by passing in struct field values that map to
// the floating IP attributes you want to see returned. SortKey allows you to
// sort by a particular network attribute. SortDir sets the direction, and is
// either `asc' or `desc'. Marker and Limit are used for pagination.
type ListOpts struct {
	Status       string `q:"status"`
	LBMethod     string `q:"lb_method"`
	Protocol     string `q:"protocol"`
	SubnetID     string `q:"subnet_id"`
	TenantID     string `q:"tenant_id"`
	AdminStateUp *bool  `q:"admin_state_up"`
	Name         string `q:"name"`
	ID           string `q:"id"`
	VIPID        string `q:"vip_id"`
	Limit        int    `q:"limit"`
	Marker       string `q:"marker"`
	SortKey      string `q:"sort_key"`
	SortDir      string `q:"sort_dir"`
}

// List returns a Pager which allows you to iterate over a collection of
// pools. It accepts a ListOpts struct, which allows you to filter and sort
// the returned collection for greater efficiency.
//
// Default policy settings return only those pools that are owned by the
// tenant who submits the request, unless an admin user submits the request.
func List(c *gophercloud.ServiceClient, opts ListOpts) pagination.Pager {
	q, err := gophercloud.BuildQueryString(&opts)
	if err != nil {
		return pagination.Pager{Err: err}
	}
	u := rootURL(c) + q.String()
	return pagination.NewPager(c, u, func(r pagination.PageResult) pagination.Page {
		return PoolPage{pagination.LinkedPageBase{PageResult: r}}
	})
}

// Supported attributes for create/update operations.
const (
	LBMethodRoundRobin       = "ROUND_ROBIN"
	LBMethodLeastConnections = "LEAST_CONNECTIONS"

	ProtocolTCP   = "TCP"
	ProtocolHTTP  = "HTTP"
	ProtocolHTTPS = "HTTPS"
)

// CreateOpts contains all the values needed to create a new pool.
type CreateOpts struct {
	// Only required if the caller has an admin role and wants to create a pool
	// for another tenant.
	TenantID string

	// Required. Name of the pool.
	Name string

	// Required. The protocol used by the pool members, you can use either
	// ProtocolTCP, ProtocolHTTP, or ProtocolHTTPS.
	Protocol string

	// The network on which the members of the pool will be located. Only members
	// that are on this network can be added to the pool.
	SubnetID string

	// The algorithm used to distribute load between the members of the pool. The
	// current specification supports LBMethodRoundRobin and
	// LBMethodLeastConnections as valid values for this attribute.
	LBMethod string
}

// Create accepts a CreateOpts struct and uses the values to create a new
// load balancer pool.
func Create(c *gophercloud.ServiceClient, opts CreateOpts) CreateResult {
	type pool struct {
		Name     string `json:"name"`
		TenantID string `json:"tenant_id,omitempty"`
		Protocol string `json:"protocol"`
		SubnetID string `json:"subnet_id"`
		LBMethod string `json:"lb_method"`
	}
	type request struct {
		Pool pool `json:"pool"`
	}

	reqBody := request{Pool: pool{
		Name:     opts.Name,
		TenantID: opts.TenantID,
		Protocol: opts.Protocol,
		SubnetID: opts.SubnetID,
		LBMethod: opts.LBMethod,
	}}

	var res CreateResult
	_, res.Err = c.Post(rootURL(c), reqBody, &res.Body, nil)
	return res
}

// Get retrieves a particular pool based on its unique ID.
func Get(c *gophercloud.ServiceClient, id string) GetResult {
	var res GetResult
	_, res.Err = c.Get(resourceURL(c, id), &res.Body, nil)
	return res
}

// UpdateOpts contains the values used when updating a pool.
type UpdateOpts struct {
	// Required. Name of the pool.
	Name string

	// The algorithm used to distribute load between the members of the pool. The
	// current specification supports LBMethodRoundRobin and
	// LBMethodLeastConnections as valid values for this attribute.
	LBMethod string
}

// Update allows pools to be updated.
func Update(c *gophercloud.ServiceClient, id string, opts UpdateOpts) UpdateResult {
	type pool struct {
		Name     string `json:"name,"`
		LBMethod string `json:"lb_method"`
	}
	type request struct {
		Pool pool `json:"pool"`
	}

	reqBody := request{Pool: pool{
		Name:     opts.Name,
		LBMethod: opts.LBMethod,
	}}

	// Send request to API
	var res UpdateResult
	_, res.Err = c.Put(resourceURL(c, id), reqBody, &res.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return res
}

// Delete will permanently delete a particular pool based on its unique ID.
func Delete(c *gophercloud.ServiceClient, id string) DeleteResult {
	var res DeleteResult
	_, res.Err = c.Delete(resourceURL(c, id), nil)
	return res
}

// AssociateMonitor will associate a health monitor with a particular pool.
// Once associated, the health monitor will start monitoring the members of the
// pool and will deactivate these members if they are deemed unhealthy. A
// member can be deactivated (status set to INACTIVE) if any of health monitors
// finds it unhealthy.
func AssociateMonitor(c *gophercloud.ServiceClient, poolID, monitorID string) AssociateResult {
	type hm struct {
		ID string `json:"id"`
	}
	type request struct {
		Monitor hm `json:"health_monitor"`
	}

	reqBody := request{hm{ID: monitorID}}

	var res AssociateResult
	_, res.Err = c.Post(associateURL(c, poolID), reqBody, &res.Body, nil)
	return res
}

// DisassociateMonitor will disassociate a health monitor with a particular
// pool. When dissociation is successful, the health monitor will no longer
// check for the health of the members of the pool.
func DisassociateMonitor(c *gophercloud.ServiceClient, poolID, monitorID string) AssociateResult {
	var res AssociateResult
	_, res.Err = c.Delete(disassociateURL(c, poolID, monitorID), nil)
	return res
}
