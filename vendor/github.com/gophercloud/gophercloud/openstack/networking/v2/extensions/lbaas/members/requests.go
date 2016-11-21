package members

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
	Weight       int    `q:"weight"`
	AdminStateUp *bool  `q:"admin_state_up"`
	TenantID     string `q:"tenant_id"`
	PoolID       string `q:"pool_id"`
	Address      string `q:"address"`
	ProtocolPort int    `q:"protocol_port"`
	ID           string `q:"id"`
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
		return MemberPage{pagination.LinkedPageBase{PageResult: r}}
	})
}

type CreateOptsBuilder interface {
	ToLBMemberCreateMap() (map[string]interface{}, error)
}

// CreateOpts contains all the values needed to create a new pool member.
type CreateOpts struct {
	// The IP address of the member.
	Address string `json:"address" required:"true"`
	// The port on which the application is hosted.
	ProtocolPort int `json:"protocol_port" required:"true"`
	// The pool to which this member will belong.
	PoolID string `json:"pool_id" required:"true"`
	// Only required if the caller has an admin role and wants to create a pool
	// for another tenant.
	TenantID string `json:"tenant_id,omitempty"`
}

func (opts CreateOpts) ToLBMemberCreateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "member")
}

// Create accepts a CreateOpts struct and uses the values to create a new
// load balancer pool member.
func Create(c *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	b, err := opts.ToLBMemberCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = c.Post(rootURL(c), b, &r.Body, nil)
	return
}

// Get retrieves a particular pool member based on its unique ID.
func Get(c *gophercloud.ServiceClient, id string) (r GetResult) {
	_, r.Err = c.Get(resourceURL(c, id), &r.Body, nil)
	return
}

type UpdateOptsBuilder interface {
	ToLBMemberUpdateMap() (map[string]interface{}, error)
}

// UpdateOpts contains the values used when updating a pool member.
type UpdateOpts struct {
	// The administrative state of the member, which is up (true) or down (false).
	AdminStateUp *bool `json:"admin_state_up,omitempty"`
}

func (opts UpdateOpts) ToLBMemberUpdateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "member")
}

// Update allows members to be updated.
func Update(c *gophercloud.ServiceClient, id string, opts UpdateOptsBuilder) (r UpdateResult) {
	b, err := opts.ToLBMemberUpdateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = c.Put(resourceURL(c, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200, 201, 202},
	})
	return
}

// Delete will permanently delete a particular member based on its unique ID.
func Delete(c *gophercloud.ServiceClient, id string) (r DeleteResult) {
	_, r.Err = c.Delete(resourceURL(c, id), nil)
	return
}
