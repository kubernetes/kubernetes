package routers

import (
	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// GatewayInfo represents the information of an external gateway for any
// particular network router.
type GatewayInfo struct {
	NetworkID string `json:"network_id" mapstructure:"network_id"`
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
	Status string `json:"status" mapstructure:"status"`

	// Information on external gateway for the router.
	GatewayInfo GatewayInfo `json:"external_gateway_info" mapstructure:"external_gateway_info"`

	// Administrative state of the router.
	AdminStateUp bool `json:"admin_state_up" mapstructure:"admin_state_up"`

	// Human readable name for the router. Does not have to be unique.
	Name string `json:"name" mapstructure:"name"`

	// Unique identifier for the router.
	ID string `json:"id" mapstructure:"id"`

	// Owner of the router. Only admin users can specify a tenant identifier
	// other than its own.
	TenantID string `json:"tenant_id" mapstructure:"tenant_id"`
}

// RouterPage is the page returned by a pager when traversing over a
// collection of routers.
type RouterPage struct {
	pagination.LinkedPageBase
}

// NextPageURL is invoked when a paginated collection of routers has reached
// the end of a page and the pager seeks to traverse over a new one. In order
// to do this, it needs to construct the next page's URL.
func (p RouterPage) NextPageURL() (string, error) {
	type resp struct {
		Links []gophercloud.Link `mapstructure:"routers_links"`
	}

	var r resp
	err := mapstructure.Decode(p.Body, &r)
	if err != nil {
		return "", err
	}

	return gophercloud.ExtractNextURL(r.Links)
}

// IsEmpty checks whether a RouterPage struct is empty.
func (p RouterPage) IsEmpty() (bool, error) {
	is, err := ExtractRouters(p)
	if err != nil {
		return true, nil
	}
	return len(is) == 0, nil
}

// ExtractRouters accepts a Page struct, specifically a RouterPage struct,
// and extracts the elements into a slice of Router structs. In other words,
// a generic collection is mapped into a relevant slice.
func ExtractRouters(page pagination.Page) ([]Router, error) {
	var resp struct {
		Routers []Router `mapstructure:"routers" json:"routers"`
	}

	err := mapstructure.Decode(page.(RouterPage).Body, &resp)

	return resp.Routers, err
}

type commonResult struct {
	gophercloud.Result
}

// Extract is a function that accepts a result and extracts a router.
func (r commonResult) Extract() (*Router, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var res struct {
		Router *Router `json:"router"`
	}

	err := mapstructure.Decode(r.Body, &res)

	return res.Router, err
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
	SubnetID string `json:"subnet_id" mapstructure:"subnet_id"`

	// The ID of the port that is a part of the subnet.
	PortID string `json:"port_id" mapstructure:"port_id"`

	// The UUID of the interface.
	ID string `json:"id" mapstructure:"id"`

	// Owner of the interface.
	TenantID string `json:"tenant_id" mapstructure:"tenant_id"`
}

// InterfaceResult represents the result of interface operations, such as
// AddInterface() and RemoveInterface().
type InterfaceResult struct {
	gophercloud.Result
}

// Extract is a function that accepts a result and extracts an information struct.
func (r InterfaceResult) Extract() (*InterfaceInfo, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var res *InterfaceInfo
	err := mapstructure.Decode(r.Body, &res)

	return res, err
}
