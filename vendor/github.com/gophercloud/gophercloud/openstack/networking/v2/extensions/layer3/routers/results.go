package routers

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// GatewayInfo represents the information of an external gateway for any
// particular network router.
type GatewayInfo struct {
	NetworkID string `json:"network_id"`
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
	// Indicates whether or not a router is currently operational.
	Status string `json:"status"`

	// Information on external gateway for the router.
	GatewayInfo GatewayInfo `json:"external_gateway_info"`

	// Administrative state of the router.
	AdminStateUp bool `json:"admin_state_up"`

	// Whether router is disitrubted or not..
	Distributed bool `json:"distributed"`

	// Human readable name for the router. Does not have to be unique.
	Name string `json:"name"`

	// Unique identifier for the router.
	ID string `json:"id"`

	// Owner of the router. Only admin users can specify a tenant identifier
	// other than its own.
	TenantID string `json:"tenant_id"`

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

// InterfaceInfo represents information about a particular router interface. As
// mentioned above, in order for a router to forward to a subnet, it needs an
// interface.
type InterfaceInfo struct {
	// The ID of the subnet which this interface is associated with.
	SubnetID string `json:"subnet_id"`

	// The ID of the port that is a part of the subnet.
	PortID string `json:"port_id"`

	// The UUID of the interface.
	ID string `json:"id"`

	// Owner of the interface.
	TenantID string `json:"tenant_id"`
}

// InterfaceResult represents the result of interface operations, such as
// AddInterface() and RemoveInterface().
type InterfaceResult struct {
	gophercloud.Result
}

// Extract is a function that accepts a result and extracts an information struct.
func (r InterfaceResult) Extract() (*InterfaceInfo, error) {
	var s InterfaceInfo
	err := r.ExtractInto(&s)
	return &s, err
}
