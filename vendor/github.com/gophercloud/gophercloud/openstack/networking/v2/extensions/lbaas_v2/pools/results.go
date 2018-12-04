package pools

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/lbaas_v2/monitors"
	"github.com/gophercloud/gophercloud/pagination"
)

// SessionPersistence represents the session persistence feature of the load
// balancing service. It attempts to force connections or requests in the same
// session to be processed by the same member as long as it is ative. Three
// types of persistence are supported:
//
// SOURCE_IP:   With this mode, all connections originating from the same source
//              IP address, will be handled by the same Member of the Pool.
// HTTP_COOKIE: With this persistence mode, the load balancing function will
//              create a cookie on the first request from a client. Subsequent
//              requests containing the same cookie value will be handled by
//              the same Member of the Pool.
// APP_COOKIE:  With this persistence mode, the load balancing function will
//              rely on a cookie established by the backend application. All
//              requests carrying the same cookie value will be handled by the
//              same Member of the Pool.
type SessionPersistence struct {
	// The type of persistence mode.
	Type string `json:"type"`

	// Name of cookie if persistence mode is set appropriately.
	CookieName string `json:"cookie_name,omitempty"`
}

// LoadBalancerID represents a load balancer.
type LoadBalancerID struct {
	ID string `json:"id"`
}

// ListenerID represents a listener.
type ListenerID struct {
	ID string `json:"id"`
}

// Pool represents a logical set of devices, such as web servers, that you
// group together to receive and process traffic. The load balancing function
// chooses a Member of the Pool according to the configured load balancing
// method to handle the new requests or connections received on the VIP address.
type Pool struct {
	// The load-balancer algorithm, which is round-robin, least-connections, and
	// so on. This value, which must be supported, is dependent on the provider.
	// Round-robin must be supported.
	LBMethod string `json:"lb_algorithm"`

	// The protocol of the Pool, which is TCP, HTTP, or HTTPS.
	Protocol string `json:"protocol"`

	// Description for the Pool.
	Description string `json:"description"`

	// A list of listeners objects IDs.
	Listeners []ListenerID `json:"listeners"` //[]map[string]interface{}

	// A list of member objects IDs.
	Members []Member `json:"members"`

	// The ID of associated health monitor.
	MonitorID string `json:"healthmonitor_id"`

	// The network on which the members of the Pool will be located. Only members
	// that are on this network can be added to the Pool.
	SubnetID string `json:"subnet_id"`

	// Owner of the Pool.
	TenantID string `json:"tenant_id"`

	// The administrative state of the Pool, which is up (true) or down (false).
	AdminStateUp bool `json:"admin_state_up"`

	// Pool name. Does not have to be unique.
	Name string `json:"name"`

	// The unique ID for the Pool.
	ID string `json:"id"`

	// A list of load balancer objects IDs.
	Loadbalancers []LoadBalancerID `json:"loadbalancers"`

	// Indicates whether connections in the same session will be processed by the
	// same Pool member or not.
	Persistence SessionPersistence `json:"session_persistence"`

	// The load balancer provider.
	Provider string `json:"provider"`

	// The Monitor associated with this Pool.
	Monitor monitors.Monitor `json:"healthmonitor"`

	// The provisioning status of the pool.
	// This value is ACTIVE, PENDING_* or ERROR.
	ProvisioningStatus string `json:"provisioning_status"`
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
// and extracts the elements into a slice of Pool structs. In other words,
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

// Extract is a function that accepts a result and extracts a pool.
func (r commonResult) Extract() (*Pool, error) {
	var s struct {
		Pool *Pool `json:"pool"`
	}
	err := r.ExtractInto(&s)
	return s.Pool, err
}

// CreateResult represents the result of a Create operation. Call its Extract
// method to interpret the result as a Pool.
type CreateResult struct {
	commonResult
}

// GetResult represents the result of a Get operation. Call its Extract
// method to interpret the result as a Pool.
type GetResult struct {
	commonResult
}

// UpdateResult represents the result of an Update operation. Call its Extract
// method to interpret the result as a Pool.
type UpdateResult struct {
	commonResult
}

// DeleteResult represents the result of a Delete operation. Call its
// ExtractErr method to determine if the request succeeded or failed.
type DeleteResult struct {
	gophercloud.ErrResult
}

// Member represents the application running on a backend server.
type Member struct {
	// Name of the Member.
	Name string `json:"name"`

	// Weight of Member.
	Weight int `json:"weight"`

	// The administrative state of the member, which is up (true) or down (false).
	AdminStateUp bool `json:"admin_state_up"`

	// Owner of the Member.
	TenantID string `json:"tenant_id"`

	// Parameter value for the subnet UUID.
	SubnetID string `json:"subnet_id"`

	// The Pool to which the Member belongs.
	PoolID string `json:"pool_id"`

	// The IP address of the Member.
	Address string `json:"address"`

	// The port on which the application is hosted.
	ProtocolPort int `json:"protocol_port"`

	// The unique ID for the Member.
	ID string `json:"id"`

	// The provisioning status of the member.
	// This value is ACTIVE, PENDING_* or ERROR.
	ProvisioningStatus string `json:"provisioning_status"`
}

// MemberPage is the page returned by a pager when traversing over a
// collection of Members in a Pool.
type MemberPage struct {
	pagination.LinkedPageBase
}

// NextPageURL is invoked when a paginated collection of members has reached
// the end of a page and the pager seeks to traverse over a new one. In order
// to do this, it needs to construct the next page's URL.
func (r MemberPage) NextPageURL() (string, error) {
	var s struct {
		Links []gophercloud.Link `json:"members_links"`
	}
	err := r.ExtractInto(&s)
	if err != nil {
		return "", err
	}
	return gophercloud.ExtractNextURL(s.Links)
}

// IsEmpty checks whether a MemberPage struct is empty.
func (r MemberPage) IsEmpty() (bool, error) {
	is, err := ExtractMembers(r)
	return len(is) == 0, err
}

// ExtractMembers accepts a Page struct, specifically a MemberPage struct,
// and extracts the elements into a slice of Members structs. In other words,
// a generic collection is mapped into a relevant slice.
func ExtractMembers(r pagination.Page) ([]Member, error) {
	var s struct {
		Members []Member `json:"members"`
	}
	err := (r.(MemberPage)).ExtractInto(&s)
	return s.Members, err
}

type commonMemberResult struct {
	gophercloud.Result
}

// ExtractMember is a function that accepts a result and extracts a member.
func (r commonMemberResult) Extract() (*Member, error) {
	var s struct {
		Member *Member `json:"member"`
	}
	err := r.ExtractInto(&s)
	return s.Member, err
}

// CreateMemberResult represents the result of a CreateMember operation.
// Call its Extract method to interpret it as a Member.
type CreateMemberResult struct {
	commonMemberResult
}

// GetMemberResult represents the result of a GetMember operation.
// Call its Extract method to interpret it as a Member.
type GetMemberResult struct {
	commonMemberResult
}

// UpdateMemberResult represents the result of an UpdateMember operation.
// Call its Extract method to interpret it as a Member.
type UpdateMemberResult struct {
	commonMemberResult
}

// DeleteMemberResult represents the result of a DeleteMember operation.
// Call its ExtractErr method to determine if the request succeeded or failed.
type DeleteMemberResult struct {
	gophercloud.ErrResult
}
