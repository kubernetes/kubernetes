package pools

import (
	"fmt"
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// AdminState gives users a solid type to work with for create and update
// operations. It is recommended that users use the `Up` and `Down` enums.
type AdminState *bool

type poolOpts struct {
	// Only required if the caller has an admin role and wants to create a pool
	// for another tenant.
	TenantID string

	// Optional. Name of the pool.
	Name string

	// Optional. Human-readable description for the pool.
	Description string

	// Required. The protocol used by the pool members, you can use either
	// ProtocolTCP, ProtocolHTTP, or ProtocolHTTPS.
	Protocol Protocol

	// The Loadbalancer on which the members of the pool will be associated with.
	// Note:  one of LoadbalancerID or ListenerID must be provided.
	LoadbalancerID string

	// The Listener on which the members of the pool will be associated with.
	// Note:  one of LoadbalancerID or ListenerID must be provided.
	ListenerID string

	// Required. The algorithm used to distribute load between the members of the pool. The
	// current specification supports LBMethodRoundRobin, LBMethodLeastConnections
	// and LBMethodSourceIp as valid values for this attribute.
	LBMethod LBMethod

	// Optional. Omit this field to prevent session persistence.
	Persistence *SessionPersistence

	// Optional. The administrative state of the Pool. A valid value is true (UP)
	// or false (DOWN).
	AdminStateUp *bool
}

// Convenience vars for AdminStateUp values.
var (
	iTrue  = true
	iFalse = false

	Up   AdminState = &iTrue
	Down AdminState = &iFalse
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
	if err != nil {
		return "", err
	}
	return q.String(), nil
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

var (
	errLoadbalancerOrListenerRequired = fmt.Errorf("A ListenerID or LoadbalancerID is required")
	errValidLBMethodRequired          = fmt.Errorf("A valid LBMethod is required. Supported values are ROUND_ROBIN, LEAST_CONNECTIONS, SOURCE_IP")
	errValidProtocolRequired          = fmt.Errorf("A valid Protocol is required. Supported values are TCP, HTTP, HTTPS")
)

// CreateOptsBuilder is the interface options structs have to satisfy in order
// to be used in the main Create operation in this package. Since many
// extensions decorate or modify the common logic, it is useful for them to
// satisfy a basic interface in order for them to be used.
type CreateOptsBuilder interface {
	ToPoolCreateMap() (map[string]interface{}, error)
}

// CreateOpts is the common options struct used in this package's Create
// operation.
type CreateOpts poolOpts

// ToPoolCreateMap casts a CreateOpts struct to a map.
func (opts CreateOpts) ToPoolCreateMap() (map[string]interface{}, error) {
	l := make(map[string]interface{})
	allowedLBMethod := map[LBMethod]bool{LBMethodRoundRobin: true, LBMethodLeastConnections: true, LBMethodSourceIp: true}
	allowedProtocol := map[Protocol]bool{ProtocolTCP: true, ProtocolHTTP: true, ProtocolHTTPS: true}

	if allowedLBMethod[opts.LBMethod] {
		l["lb_algorithm"] = opts.LBMethod
	} else {
		return nil, errValidLBMethodRequired
	}
	if allowedProtocol[opts.Protocol] {
		l["protocol"] = opts.Protocol
	} else {
		return nil, errValidProtocolRequired
	}
	if opts.LoadbalancerID == "" && opts.ListenerID == "" {
		return nil, errLoadbalancerOrListenerRequired
	} else {
		if opts.LoadbalancerID != "" {
			l["loadbalancer_id"] = opts.LoadbalancerID
		}
		if opts.ListenerID != "" {
			l["listener_id"] = opts.ListenerID
		}
	}
	if opts.AdminStateUp != nil {
		l["admin_state_up"] = &opts.AdminStateUp
	}
	if opts.Name != "" {
		l["name"] = opts.Name
	}
	if opts.TenantID != "" {
		l["tenant_id"] = opts.TenantID
	}
	if opts.Persistence != nil {
		l["session_persistence"] = &opts.Persistence
	}

	return map[string]interface{}{"pool": l}, nil
}

// Create accepts a CreateOpts struct and uses the values to create a new
// load balancer pool.
func Create(c *gophercloud.ServiceClient, opts CreateOpts) CreateResult {
	var res CreateResult

	reqBody, err := opts.ToPoolCreateMap()
	if err != nil {
		res.Err = err
		return res
	}

	// Send request to API
	_, res.Err = c.Post(rootURL(c), reqBody, &res.Body, nil)
	return res
}

// Get retrieves a particular pool based on its unique ID.
func Get(c *gophercloud.ServiceClient, id string) GetResult {
	var res GetResult
	_, res.Err = c.Get(resourceURL(c, id), &res.Body, nil)
	return res
}

// UpdateOptsBuilder is the interface options structs have to satisfy in order
// to be used in the main Update operation in this package. Since many
// extensions decorate or modify the common logic, it is useful for them to
// satisfy a basic interface in order for them to be used.
type UpdateOptsBuilder interface {
	ToPoolUpdateMap() (map[string]interface{}, error)
}

// UpdateOpts is the common options struct used in this package's Update
// operation.
type UpdateOpts poolOpts

// ToPoolUpdateMap casts a UpdateOpts struct to a map.
func (opts UpdateOpts) ToPoolUpdateMap() (map[string]interface{}, error) {
	l := make(map[string]interface{})
	allowedLBMethod := map[LBMethod]bool{LBMethodRoundRobin: true, LBMethodLeastConnections: true, LBMethodSourceIp: true}

	if opts.LBMethod != "" {
		if allowedLBMethod[opts.LBMethod] {
			l["lb_algorithm"] = opts.LBMethod
		} else {
			return nil, errValidLBMethodRequired
		}
	}
	if opts.Name != "" {
		l["name"] = opts.Name
	}
	if opts.Description != "" {
		l["description"] = opts.Description
	}
	if opts.AdminStateUp != nil {
		l["admin_state_up"] = &opts.AdminStateUp
	}

	return map[string]interface{}{"pool": l}, nil
}

// Update allows pools to be updated.
func Update(c *gophercloud.ServiceClient, id string, opts UpdateOptsBuilder) UpdateResult {
	var res UpdateResult

	reqBody, err := opts.ToPoolUpdateMap()
	if err != nil {
		res.Err = err
		return res
	}

	// Send request to API
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

// CreateOpts contains all the values needed to create a new Member for a Pool.
type memberOpts struct {
	// Optional. Name of the Member.
	Name string

	// Only required if the caller has an admin role and wants to create a Member
	// for another tenant.
	TenantID string

	// Required. The IP address of the member to receive traffic from the load balancer.
	Address string

	// Required. The port on which to listen for client traffic.
	ProtocolPort int

	// Optional. A positive integer value that indicates the relative portion of
	// traffic that this member should receive from the pool. For example, a
	// member with a weight of 10 receives five times as much traffic as a member
	// with a weight of 2.
	Weight int

	// Optional.  If you omit this parameter, LBaaS uses the vip_subnet_id
	// parameter value for the subnet UUID.
	SubnetID string

	// Optional. The administrative state of the Pool. A valid value is true (UP)
	// or false (DOWN).
	AdminStateUp *bool
}

// MemberListOptsBuilder allows extensions to add additional parameters to the
// Member List request.
type MemberListOptsBuilder interface {
	ToMemberListQuery() (string, error)
}

// MemberListOpts allows the filtering and sorting of paginated collections through
// the API. Filtering is achieved by passing in struct field values that map to
// the Member attributes you want to see returned. SortKey allows you to
// sort by a particular Member attribute. SortDir sets the direction, and is
// either `asc' or `desc'. Marker and Limit are used for pagination.
type MemberListOpts struct {
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
func (opts MemberListOpts) ToMemberListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	if err != nil {
		return "", err
	}
	return q.String(), nil
}

// List returns a Pager which allows you to iterate over a collection of
// members. It accepts a ListOpts struct, which allows you to filter and sort
// the returned collection for greater efficiency.
//
// Default policy settings return only those members that are owned by the
// tenant who submits the request, unless an admin user submits the request.
func ListAssociateMembers(c *gophercloud.ServiceClient, poolID string, opts MemberListOptsBuilder) pagination.Pager {
	url := memberRootURL(c, poolID)
	if opts != nil {
		query, err := opts.ToMemberListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}

	return pagination.NewPager(c, url, func(r pagination.PageResult) pagination.Page {
		return MemberPage{pagination.LinkedPageBase{PageResult: r}}
	})
}

var (
	errPoolIdRequired       = fmt.Errorf("PoolID is required")
	errAddressRequired      = fmt.Errorf("Address is required")
	errProtocolPortRequired = fmt.Errorf("ProtocolPort is required")
)

// MemberCreateOptsBuilder is the interface options structs have to satisfy in order
// to be used in the main Create operation in this package. Since many
// extensions decorate or modify the common logic, it is useful for them to
// satisfy a basic interface in order for them to be used.
type MemberCreateOptsBuilder interface {
	ToMemberCreateMap() (map[string]interface{}, error)
}

// MemberCreateOpts is the common options struct used in this package's Create
// operation.
type MemberCreateOpts memberOpts

// ToMemberCreateMap casts a CreateOpts struct to a map.
func (opts MemberCreateOpts) ToMemberCreateMap() (map[string]interface{}, error) {
	l := make(map[string]interface{})

	if opts.Address != "" {
		l["address"] = opts.Address
	} else {
		return nil, errAddressRequired
	}
	if opts.ProtocolPort != 0 {
		l["protocol_port"] = opts.ProtocolPort
	} else {
		return nil, errProtocolPortRequired
	}
	if opts.AdminStateUp != nil {
		l["admin_state_up"] = &opts.AdminStateUp
	}
	if opts.Name != "" {
		l["name"] = opts.Name
	}
	if opts.TenantID != "" {
		l["tenant_id"] = opts.TenantID
	}
	if opts.SubnetID != "" {
		l["subnet_id"] = opts.SubnetID
	}
	if opts.Weight != 0 {
		l["weight"] = opts.Weight
	}

	return map[string]interface{}{"member": l}, nil
}

// CreateAssociateMember will create and associate a Member with a particular Pool.
func CreateAssociateMember(c *gophercloud.ServiceClient, poolID string, opts MemberCreateOpts) AssociateResult {
	var res AssociateResult

	if poolID == "" {
		res.Err = errPoolIdRequired
		return res
	}

	reqBody, err := opts.ToMemberCreateMap()
	if err != nil {
		res.Err = err
		return res
	}

	_, res.Err = c.Post(memberRootURL(c, poolID), reqBody, &res.Body, nil)
	return res
}

// Get retrieves a particular Pool Member based on its unique ID.
func GetAssociateMember(c *gophercloud.ServiceClient, poolID string, memberID string) GetResult {
	var res GetResult
	_, res.Err = c.Get(memberResourceURL(c, poolID, memberID), &res.Body, nil)
	return res
}

// MemberUpdateOptsBuilder is the interface options structs have to satisfy in order
// to be used in the main Update operation in this package. Since many
// extensions decorate or modify the common logic, it is useful for them to
// satisfy a basic interface in order for them to be used.
type MemberUpdateOptsBuilder interface {
	ToMemberUpdateMap() (map[string]interface{}, error)
}

// UpdateOpts is the common options struct used in this package's Update
// operation.
type MemberUpdateOpts memberOpts

// ToMemberUpdateMap casts a UpdateOpts struct to a map.
func (opts MemberUpdateOpts) ToMemberUpdateMap() (map[string]interface{}, error) {
	l := make(map[string]interface{})

	if opts.AdminStateUp != nil {
		l["admin_state_up"] = &opts.AdminStateUp
	}
	if opts.Name != "" {
		l["name"] = opts.Name
	}
	if opts.Weight != 0 {
		l["weight"] = opts.Weight
	}

	return map[string]interface{}{"member": l}, nil
}

// Update allows Member to be updated.
func UpdateAssociateMember(c *gophercloud.ServiceClient, poolID string, memberID string, opts MemberUpdateOpts) UpdateResult {
	var res UpdateResult

	reqBody, err := opts.ToMemberUpdateMap()
	if err != nil {
		res.Err = err
		return res
	}

	// Send request to API
	_, res.Err = c.Put(memberResourceURL(c, poolID, memberID), reqBody, &res.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200, 201, 202},
	})
	return res
}

// DisassociateMember will remove and disassociate a Member from a particular Pool.
func DeleteMember(c *gophercloud.ServiceClient, poolID string, memberID string) DeleteResult {
	var res DeleteResult
	_, res.Err = c.Delete(memberResourceURL(c, poolID, memberID), nil)
	return res
}
