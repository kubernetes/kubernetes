package vips

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
	ID              string `q:"id"`
	Name            string `q:"name"`
	AdminStateUp    *bool  `q:"admin_state_up"`
	Status          string `q:"status"`
	TenantID        string `q:"tenant_id"`
	SubnetID        string `q:"subnet_id"`
	Address         string `q:"address"`
	PortID          string `q:"port_id"`
	Protocol        string `q:"protocol"`
	ProtocolPort    int    `q:"protocol_port"`
	ConnectionLimit int    `q:"connection_limit"`
	Limit           int    `q:"limit"`
	Marker          string `q:"marker"`
	SortKey         string `q:"sort_key"`
	SortDir         string `q:"sort_dir"`
}

// List returns a Pager which allows you to iterate over a collection of
// Virtual IPs. It accepts a ListOpts struct, which allows you to filter and
// sort the returned collection for greater efficiency.
//
// Default policy settings return only those virtual IPs that are owned by the
// tenant who submits the request, unless an admin user submits the request.
func List(c *gophercloud.ServiceClient, opts ListOpts) pagination.Pager {
	q, err := gophercloud.BuildQueryString(&opts)
	if err != nil {
		return pagination.Pager{Err: err}
	}
	u := rootURL(c) + q.String()
	return pagination.NewPager(c, u, func(r pagination.PageResult) pagination.Page {
		return VIPPage{pagination.LinkedPageBase{PageResult: r}}
	})
}

// CreateOptsBuilder allows extensions to add additional parameters to the
// Create Request.
type CreateOptsBuilder interface {
	ToVIPCreateMap() (map[string]interface{}, error)
}

// CreateOpts contains all the values needed to create a new virtual IP.
type CreateOpts struct {
	// Name is the human-readable name for the VIP. Does not have to be unique.
	Name string `json:"name" required:"true"`

	// SubnetID is the network on which to allocate the VIP's address. A tenant
	// can only create VIPs on networks authorized by policy (e.g. networks that
	// belong to them or networks that are shared).
	SubnetID string `json:"subnet_id" required:"true"`

	// Protocol - can either be TCP, HTTP or HTTPS.
	Protocol string `json:"protocol" required:"true"`

	// ProtocolPort is the port on which to listen for client traffic.
	ProtocolPort int `json:"protocol_port" required:"true"`

	// PoolID is the ID of the pool with which the VIP is associated.
	PoolID string `json:"pool_id" required:"true"`

	// TenantID is only required if the caller has an admin role and wants
	// to create a pool for another tenant.
	TenantID string `json:"tenant_id,omitempty"`

	// Address is the IP address of the VIP.
	Address string `json:"address,omitempty"`

	// Description is the human-readable description for the VIP.
	Description string `json:"description,omitempty"`

	// Persistence is the the of session persistence to use.
	// Omit this field to prevent session persistence.
	Persistence *SessionPersistence `json:"session_persistence,omitempty"`

	// ConnLimit is the maximum number of connections allowed for the VIP.
	ConnLimit *int `json:"connection_limit,omitempty"`

	// AdminStateUp is the administrative state of the VIP. A valid value is
	// true (UP) or false (DOWN).
	AdminStateUp *bool `json:"admin_state_up,omitempty"`
}

// ToVIPCreateMap builds a request body from CreateOpts.
func (opts CreateOpts) ToVIPCreateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "vip")
}

// Create is an operation which provisions a new virtual IP based on the
// configuration defined in the CreateOpts struct. Once the request is
// validated and progress has started on the provisioning process, a
// CreateResult will be returned.
//
// Please note that the PoolID should refer to a pool that is not already
// associated with another vip. If the pool is already used by another vip,
// then the operation will fail with a 409 Conflict error will be returned.
//
// Users with an admin role can create VIPs on behalf of other tenants by
// specifying a TenantID attribute different than their own.
func Create(c *gophercloud.ServiceClient, opts CreateOpts) (r CreateResult) {
	b, err := opts.ToVIPCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = c.Post(rootURL(c), b, &r.Body, nil)
	return
}

// Get retrieves a particular virtual IP based on its unique ID.
func Get(c *gophercloud.ServiceClient, id string) (r GetResult) {
	_, r.Err = c.Get(resourceURL(c, id), &r.Body, nil)
	return
}

// UpdateOptsBuilder allows extensions to add additional parameters to the
// Update request.
type UpdateOptsBuilder interface {
	ToVIPUpdateMap() (map[string]interface{}, error)
}

// UpdateOpts contains all the values needed to update an existing virtual IP.
// Attributes not listed here but appear in CreateOpts are immutable and cannot
// be updated.
type UpdateOpts struct {
	// Name is the human-readable name for the VIP. Does not have to be unique.
	Name *string `json:"name,omitempty"`

	// PoolID is the ID of the pool with which the VIP is associated.
	PoolID *string `json:"pool_id,omitempty"`

	// Description is the human-readable description for the VIP.
	Description *string `json:"description,omitempty"`

	// Persistence is the the of session persistence to use.
	// Omit this field to prevent session persistence.
	Persistence *SessionPersistence `json:"session_persistence,omitempty"`

	// ConnLimit is the maximum number of connections allowed for the VIP.
	ConnLimit *int `json:"connection_limit,omitempty"`

	// AdminStateUp is the administrative state of the VIP. A valid value is
	// true (UP) or false (DOWN).
	AdminStateUp *bool `json:"admin_state_up,omitempty"`
}

// ToVIPUpdateMap builds a request body based on UpdateOpts.
func (opts UpdateOpts) ToVIPUpdateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "vip")
}

// Update is an operation which modifies the attributes of the specified VIP.
func Update(c *gophercloud.ServiceClient, id string, opts UpdateOptsBuilder) (r UpdateResult) {
	b, err := opts.ToVIPUpdateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = c.Put(resourceURL(c, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200, 202},
	})
	return
}

// Delete will permanently delete a particular virtual IP based on its unique ID.
func Delete(c *gophercloud.ServiceClient, id string) (r DeleteResult) {
	_, r.Err = c.Delete(resourceURL(c, id), nil)
	return
}
