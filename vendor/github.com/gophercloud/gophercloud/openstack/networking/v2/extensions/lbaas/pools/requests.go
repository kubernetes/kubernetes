package pools

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
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

// LBMethod is a type used for possible load balancing methods.
type LBMethod string

// LBProtocol is a type used for possible load balancing protocols.
type LBProtocol string

// Supported attributes for create/update operations.
const (
	LBMethodRoundRobin       LBMethod = "ROUND_ROBIN"
	LBMethodLeastConnections LBMethod = "LEAST_CONNECTIONS"

	ProtocolTCP   LBProtocol = "TCP"
	ProtocolHTTP  LBProtocol = "HTTP"
	ProtocolHTTPS LBProtocol = "HTTPS"
)

// CreateOptsBuilder allows extensions to add additional parameters to the
// Create request.
type CreateOptsBuilder interface {
	ToLBPoolCreateMap() (map[string]interface{}, error)
}

// CreateOpts contains all the values needed to create a new pool.
type CreateOpts struct {
	// Name of the pool.
	Name string `json:"name" required:"true"`

	// Protocol used by the pool members, you can use either
	// ProtocolTCP, ProtocolHTTP, or ProtocolHTTPS.
	Protocol LBProtocol `json:"protocol" required:"true"`

	// TenantID is only required if the caller has an admin role and wants
	// to create a pool for another tenant.
	TenantID string `json:"tenant_id,omitempty"`

	// SubnetID is the network on which the members of the pool will be located.
	// Only members that are on this network can be added to the pool.
	SubnetID string `json:"subnet_id,omitempty"`

	// LBMethod is the algorithm used to distribute load between the members of
	// the pool. The current specification supports LBMethodRoundRobin and
	// LBMethodLeastConnections as valid values for this attribute.
	LBMethod LBMethod `json:"lb_method" required:"true"`

	// Provider of the pool.
	Provider string `json:"provider,omitempty"`
}

// ToLBPoolCreateMap builds a request body based on CreateOpts.
func (opts CreateOpts) ToLBPoolCreateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "pool")
}

// Create accepts a CreateOptsBuilder and uses the values to create a new
// load balancer pool.
func Create(c *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	b, err := opts.ToLBPoolCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = c.Post(rootURL(c), b, &r.Body, nil)
	return
}

// Get retrieves a particular pool based on its unique ID.
func Get(c *gophercloud.ServiceClient, id string) (r GetResult) {
	_, r.Err = c.Get(resourceURL(c, id), &r.Body, nil)
	return
}

// UpdateOptsBuilder allows extensions to add additional parameters ot the
// Update request.
type UpdateOptsBuilder interface {
	ToLBPoolUpdateMap() (map[string]interface{}, error)
}

// UpdateOpts contains the values used when updating a pool.
type UpdateOpts struct {
	// Name of the pool.
	Name string `json:"name,omitempty"`

	// LBMethod is the algorithm used to distribute load between the members of
	// the pool. The current specification supports LBMethodRoundRobin and
	// LBMethodLeastConnections as valid values for this attribute.
	LBMethod LBMethod `json:"lb_method,omitempty"`
}

// ToLBPoolUpdateMap builds a request body based on UpdateOpts.
func (opts UpdateOpts) ToLBPoolUpdateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "pool")
}

// Update allows pools to be updated.
func Update(c *gophercloud.ServiceClient, id string, opts UpdateOptsBuilder) (r UpdateResult) {
	b, err := opts.ToLBPoolUpdateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = c.Put(resourceURL(c, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

// Delete will permanently delete a particular pool based on its unique ID.
func Delete(c *gophercloud.ServiceClient, id string) (r DeleteResult) {
	_, r.Err = c.Delete(resourceURL(c, id), nil)
	return
}

// AssociateMonitor will associate a health monitor with a particular pool.
// Once associated, the health monitor will start monitoring the members of the
// pool and will deactivate these members if they are deemed unhealthy. A
// member can be deactivated (status set to INACTIVE) if any of health monitors
// finds it unhealthy.
func AssociateMonitor(c *gophercloud.ServiceClient, poolID, monitorID string) (r AssociateResult) {
	b := map[string]interface{}{"health_monitor": map[string]string{"id": monitorID}}
	_, r.Err = c.Post(associateURL(c, poolID), b, &r.Body, nil)
	return
}

// DisassociateMonitor will disassociate a health monitor with a particular
// pool. When dissociation is successful, the health monitor will no longer
// check for the health of the members of the pool.
func DisassociateMonitor(c *gophercloud.ServiceClient, poolID, monitorID string) (r AssociateResult) {
	_, r.Err = c.Delete(disassociateURL(c, poolID, monitorID), nil)
	return
}
