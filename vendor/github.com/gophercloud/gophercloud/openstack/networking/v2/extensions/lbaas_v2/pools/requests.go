package pools

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// ListOptsBuilder allows extensions to add additional parameters to the
// List request.
type ListOptsBuilder interface {
	ToPoolListQuery() (string, error)
}

// ListOpts allows the filtering and sorting of paginated collections through
// the API. Filtering is achieved by passing in struct field values that map to
// the Pool attributes you want to see returned. SortKey allows you to
// sort by a particular Pool attribute. SortDir sets the direction, and is
// either `asc' or `desc'. Marker and Limit are used for pagination.
type ListOpts struct {
	LBMethod       string `q:"lb_algorithm"`
	Protocol       string `q:"protocol"`
	TenantID       string `q:"tenant_id"`
	AdminStateUp   *bool  `q:"admin_state_up"`
	Name           string `q:"name"`
	ID             string `q:"id"`
	LoadbalancerID string `q:"loadbalancer_id"`
	ListenerID     string `q:"listener_id"`
	Limit          int    `q:"limit"`
	Marker         string `q:"marker"`
	SortKey        string `q:"sort_key"`
	SortDir        string `q:"sort_dir"`
}

// ToPoolListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToPoolListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// List returns a Pager which allows you to iterate over a collection of
// pools. It accepts a ListOpts struct, which allows you to filter and sort
// the returned collection for greater efficiency.
//
// Default policy settings return only those pools that are owned by the
// tenant who submits the request, unless an admin user submits the request.
func List(c *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := rootURL(c)
	if opts != nil {
		query, err := opts.ToPoolListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}
	return pagination.NewPager(c, url, func(r pagination.PageResult) pagination.Page {
		return PoolPage{pagination.LinkedPageBase{PageResult: r}}
	})
}

type LBMethod string
type Protocol string

// Supported attributes for create/update operations.
const (
	LBMethodRoundRobin       LBMethod = "ROUND_ROBIN"
	LBMethodLeastConnections LBMethod = "LEAST_CONNECTIONS"
	LBMethodSourceIp         LBMethod = "SOURCE_IP"

	ProtocolTCP   Protocol = "TCP"
	ProtocolHTTP  Protocol = "HTTP"
	ProtocolHTTPS Protocol = "HTTPS"
)

// CreateOptsBuilder allows extensions to add additional parameters to the
// Create request.
type CreateOptsBuilder interface {
	ToPoolCreateMap() (map[string]interface{}, error)
}

// CreateOpts is the common options struct used in this package's Create
// operation.
type CreateOpts struct {
	// The algorithm used to distribute load between the members of the pool. The
	// current specification supports LBMethodRoundRobin, LBMethodLeastConnections
	// and LBMethodSourceIp as valid values for this attribute.
	LBMethod LBMethod `json:"lb_algorithm" required:"true"`

	// The protocol used by the pool members, you can use either
	// ProtocolTCP, ProtocolHTTP, or ProtocolHTTPS.
	Protocol Protocol `json:"protocol" required:"true"`

	// The Loadbalancer on which the members of the pool will be associated with.
	// Note: one of LoadbalancerID or ListenerID must be provided.
	LoadbalancerID string `json:"loadbalancer_id,omitempty" xor:"ListenerID"`

	// The Listener on which the members of the pool will be associated with.
	// Note: one of LoadbalancerID or ListenerID must be provided.
	ListenerID string `json:"listener_id,omitempty" xor:"LoadbalancerID"`

	// The UUID of the tenant who owns the Pool. Only administrative users
	// can specify a tenant UUID other than their own.
	TenantID string `json:"tenant_id,omitempty"`

	// Name of the pool.
	Name string `json:"name,omitempty"`

	// Human-readable description for the pool.
	Description string `json:"description,omitempty"`

	// Persistence is the session persistence of the pool.
	// Omit this field to prevent session persistence.
	Persistence *SessionPersistence `json:"session_persistence,omitempty"`

	// The administrative state of the Pool. A valid value is true (UP)
	// or false (DOWN).
	AdminStateUp *bool `json:"admin_state_up,omitempty"`
}

// ToPoolCreateMap builds a request body from CreateOpts.
func (opts CreateOpts) ToPoolCreateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "pool")
}

// Create accepts a CreateOpts struct and uses the values to create a new
// load balancer pool.
func Create(c *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	b, err := opts.ToPoolCreateMap()
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

// UpdateOptsBuilder allows extensions to add additional parameters to the
// Update request.
type UpdateOptsBuilder interface {
	ToPoolUpdateMap() (map[string]interface{}, error)
}

// UpdateOpts is the common options struct used in this package's Update
// operation.
type UpdateOpts struct {
	// Name of the pool.
	Name string `json:"name,omitempty"`

	// Human-readable description for the pool.
	Description string `json:"description,omitempty"`

	// The algorithm used to distribute load between the members of the pool. The
	// current specification supports LBMethodRoundRobin, LBMethodLeastConnections
	// and LBMethodSourceIp as valid values for this attribute.
	LBMethod LBMethod `json:"lb_algorithm,omitempty"`

	// The administrative state of the Pool. A valid value is true (UP)
	// or false (DOWN).
	AdminStateUp *bool `json:"admin_state_up,omitempty"`
}

// ToPoolUpdateMap builds a request body from UpdateOpts.
func (opts UpdateOpts) ToPoolUpdateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "pool")
}

// Update allows pools to be updated.
func Update(c *gophercloud.ServiceClient, id string, opts UpdateOptsBuilder) (r UpdateResult) {
	b, err := opts.ToPoolUpdateMap()
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

// ListMemberOptsBuilder allows extensions to add additional parameters to the
// ListMembers request.
type ListMembersOptsBuilder interface {
	ToMembersListQuery() (string, error)
}

// ListMembersOpts allows the filtering and sorting of paginated collections
// through the API. Filtering is achieved by passing in struct field values
// that map to the Member attributes you want to see returned. SortKey allows
// you to sort by a particular Member attribute. SortDir sets the direction,
// and is either `asc' or `desc'. Marker and Limit are used for pagination.
type ListMembersOpts struct {
	Name         string `q:"name"`
	Weight       int    `q:"weight"`
	AdminStateUp *bool  `q:"admin_state_up"`
	TenantID     string `q:"tenant_id"`
	Address      string `q:"address"`
	ProtocolPort int    `q:"protocol_port"`
	ID           string `q:"id"`
	Limit        int    `q:"limit"`
	Marker       string `q:"marker"`
	SortKey      string `q:"sort_key"`
	SortDir      string `q:"sort_dir"`
}

// ToMemberListQuery formats a ListOpts into a query string.
func (opts ListMembersOpts) ToMembersListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// ListMembers returns a Pager which allows you to iterate over a collection of
// members. It accepts a ListMembersOptsBuilder, which allows you to filter and
// sort the returned collection for greater efficiency.
//
// Default policy settings return only those members that are owned by the
// tenant who submits the request, unless an admin user submits the request.
func ListMembers(c *gophercloud.ServiceClient, poolID string, opts ListMembersOptsBuilder) pagination.Pager {
	url := memberRootURL(c, poolID)
	if opts != nil {
		query, err := opts.ToMembersListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}
	return pagination.NewPager(c, url, func(r pagination.PageResult) pagination.Page {
		return MemberPage{pagination.LinkedPageBase{PageResult: r}}
	})
}

// CreateMemberOptsBuilder allows extensions to add additional parameters to the
// CreateMember request.
type CreateMemberOptsBuilder interface {
	ToMemberCreateMap() (map[string]interface{}, error)
}

// CreateMemberOpts is the common options struct used in this package's CreateMember
// operation.
type CreateMemberOpts struct {
	// The IP address of the member to receive traffic from the load balancer.
	Address string `json:"address" required:"true"`

	// The port on which to listen for client traffic.
	ProtocolPort int `json:"protocol_port" required:"true"`

	// Name of the Member.
	Name string `json:"name,omitempty"`

	// The UUID of the tenant who owns the Member. Only administrative users
	// can specify a tenant UUID other than their own.
	TenantID string `json:"tenant_id,omitempty"`

	// A positive integer value that indicates the relative portion of traffic
	// that  this member should receive from the pool. For example, a member with
	// a weight  of 10 receives five times as much traffic as a member with a
	// weight of 2.
	Weight int `json:"weight,omitempty"`

	// If you omit this parameter, LBaaS uses the vip_subnet_id parameter value
	// for the subnet UUID.
	SubnetID string `json:"subnet_id,omitempty"`

	// The administrative state of the Pool. A valid value is true (UP)
	// or false (DOWN).
	AdminStateUp *bool `json:"admin_state_up,omitempty"`
}

// ToMemberCreateMap builds a request body from CreateMemberOpts.
func (opts CreateMemberOpts) ToMemberCreateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "member")
}

// CreateMember will create and associate a Member with a particular Pool.
func CreateMember(c *gophercloud.ServiceClient, poolID string, opts CreateMemberOpts) (r CreateMemberResult) {
	b, err := opts.ToMemberCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = c.Post(memberRootURL(c, poolID), b, &r.Body, nil)
	return
}

// GetMember retrieves a particular Pool Member based on its unique ID.
func GetMember(c *gophercloud.ServiceClient, poolID string, memberID string) (r GetMemberResult) {
	_, r.Err = c.Get(memberResourceURL(c, poolID, memberID), &r.Body, nil)
	return
}

// UpdateMemberOptsBuilder allows extensions to add additional parameters to the
// List request.
type UpdateMemberOptsBuilder interface {
	ToMemberUpdateMap() (map[string]interface{}, error)
}

// UpdateMemberOpts is the common options struct used in this package's Update
// operation.
type UpdateMemberOpts struct {
	// Name of the Member.
	Name string `json:"name,omitempty"`

	// A positive integer value that indicates the relative portion of traffic
	// that this member should receive from the pool. For example, a member with
	// a weight of 10 receives five times as much traffic as a member with a
	// weight of 2.
	Weight int `json:"weight,omitempty"`

	// The administrative state of the Pool. A valid value is true (UP)
	// or false (DOWN).
	AdminStateUp *bool `json:"admin_state_up,omitempty"`
}

// ToMemberUpdateMap builds a request body from UpdateMemberOpts.
func (opts UpdateMemberOpts) ToMemberUpdateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "member")
}

// Update allows Member to be updated.
func UpdateMember(c *gophercloud.ServiceClient, poolID string, memberID string, opts UpdateMemberOptsBuilder) (r UpdateMemberResult) {
	b, err := opts.ToMemberUpdateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = c.Put(memberResourceURL(c, poolID, memberID), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200, 201, 202},
	})
	return
}

// DisassociateMember will remove and disassociate a Member from a particular
// Pool.
func DeleteMember(c *gophercloud.ServiceClient, poolID string, memberID string) (r DeleteMemberResult) {
	_, r.Err = c.Delete(memberResourceURL(c, poolID, memberID), nil)
	return
}
