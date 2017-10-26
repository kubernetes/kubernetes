package routers

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// GatewayInfo represents the information of an external gateway for any
// particular network router.
type GatewayInfo struct {
	NetworkID        string            `json:"network_id"`
	ExternalFixedIPs []ExternalFixedIP `json:"external_fixed_ips,omitempty"`
}

// ExternalFixedIP is the IP address and subnet ID of the external gateway of a
// router.
type ExternalFixedIP struct {
	IPAddress string `json:"ip_address"`
	SubnetID  string `json:"subnet_id"`
}

// Route is a possible route in a router.
type Route struct {
	NextHop         string `json:"nexthop"`
	DestinationCIDR string `json:"destination"`
}

// Router represents a Neutron router. A router is a logical entity that
// forwards packets across internal subnets and NATs (network address
// translation) them on external networks through an appropriate gateway.
//
// A router has an interface for each subnet with which it is associated. By
// default, the IP address of such interface is the subnet's gateway IP. Also,
// whenever a router is associated with a subnet, a port for that router
// interface is added to the subnet's network.
type Router struct {
	// Status indicates whether or not a router is currently operational.
	Status string `json:"status"`

	// GateayInfo provides information on external gateway for the router.
	GatewayInfo GatewayInfo `json:"external_gateway_info"`

	// AdminStateUp is the administrative state of the router.
	AdminStateUp bool `json:"admin_state_up"`

	// Distributed is whether router is disitrubted or not.
	Distributed bool `json:"distributed"`

	// Name is the human readable name for the router. It does not have to be
	// unique.
	Name string `json:"name"`

	// ID is the unique identifier for the router.
	ID string `json:"id"`

	// TenantID is the owner of the router. Only admin users can specify a tenant
	// identifier other than its own.
	TenantID string `json:"tenant_id"`

	// Routes are a collection of static routes that the router will host.
	Routes []Route `json:"routes"`
}

// RouterPage is the page returned by a pager when traversing over a
// collection of routers.
type RouterPage struct {
	pagination.LinkedPageBase
}

// NextPageURL is invoked when a paginated collection of routers has reached
// the end of a page and the pager seeks to traverse over a new one. In order
// to do this, it needs to construct the next page's URL.
func (r RouterPage) NextPageURL() (string, error) {
	var s struct {
		Links []gophercloud.Link `json:"routers_links"`
	}
	err := r.ExtractInto(&s)
	if err != nil {
		return "", err
	}
	return gophercloud.ExtractNextURL(s.Links)
}

// IsEmpty checks whether a RouterPage struct is empty.
func (r RouterPage) IsEmpty() (bool, error) {
	is, err := ExtractRouters(r)
	return len(is) == 0, err
}

// ExtractRouters accepts a Page struct, specifically a RouterPage struct,
// and extracts the elements into a slice of Router structs. In other words,
// a generic collection is mapped into a relevant slice.
func ExtractRouters(r pagination.Page) ([]Router, error) {
	var s struct {
		Routers []Router `json:"routers"`
	}
	err := (r.(RouterPage)).ExtractInto(&s)
	return s.Routers, err
}

type commonResult struct {
	gophercloud.Result
}

// Extract is a function that accepts a result and extracts a router.
func (r commonResult) Extract() (*Router, error) {
	var s struct {
		Router *Router `json:"router"`
	}
	err := r.ExtractInto(&s)
	return s.Router, err
}

// CreateResult represents the result of a create operation. Call its Extract
// method to interpret it as a Router.
type CreateResult struct {
	commonResult
}

// GetResult represents the result of a get operation. Call its Extract
// method to interpret it as a Router.
type GetResult struct {
	commonResult
}

// UpdateResult represents the result of an update operation. Call its Extract
// method to interpret it as a Router.
type UpdateResult struct {
	commonResult
}

// DeleteResult represents the result of a delete operation. Call its ExtractErr
// method to determine if the request succeeded or failed.
type DeleteResult struct {
	gophercloud.ErrResult
}

// InterfaceInfo represents information about a particular router interface. As
// mentioned above, in order for a router to forward to a subnet, it needs an
// interface.
type InterfaceInfo struct {
	// SubnetID is the ID of the subnet which this interface is associated with.
	SubnetID string `json:"subnet_id"`

	// PortID is the ID of the port that is a part of the subnet.
	PortID string `json:"port_id"`

	// ID is the UUID of the interface.
	ID string `json:"id"`

	// TenantID is the owner of the interface.
	TenantID string `json:"tenant_id"`
}

// InterfaceResult represents the result of interface operations, such as
// AddInterface() and RemoveInterface(). Call its Extract method to interpret
// the result as a InterfaceInfo.
type InterfaceResult struct {
	gophercloud.Result
}

// Extract is a function that accepts a result and extracts an information struct.
func (r InterfaceResult) Extract() (*InterfaceInfo, error) {
	var s InterfaceInfo
	err := r.ExtractInto(&s)
	return &s, err
}
