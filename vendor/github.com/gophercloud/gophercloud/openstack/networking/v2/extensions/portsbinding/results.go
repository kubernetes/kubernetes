package portsbinding

import (
	"github.com/gophercloud/gophercloud"

	"github.com/gophercloud/gophercloud/openstack/networking/v2/ports"
	"github.com/gophercloud/gophercloud/pagination"
)

type commonResult struct {
	gophercloud.Result
}

// Extract is a function that accepts a result and extracts a port resource.
func (r commonResult) Extract() (*Port, error) {
	var s struct {
		Port *Port `json:"port"`
	}
	err := r.ExtractInto(&s)
	return s.Port, err
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

// IP is a sub-struct that represents an individual IP.
type IP struct {
	SubnetID  string `json:"subnet_id"`
	IPAddress string `json:"ip_address"`
}

// Port represents a Neutron port. See package documentation for a top-level
// description of what this is.
type Port struct {
	ports.Port
	// The ID of the host where the port is allocated
	HostID string `json:"binding:host_id"`
	// A dictionary that enables the application to pass information about
	// functions that the Networking API provides.
	VIFDetails map[string]interface{} `json:"binding:vif_details"`
	// The VIF type for the port.
	VIFType string `json:"binding:vif_type"`
	// The virtual network interface card (vNIC) type that is bound to the
	// neutron port
	VNICType string `json:"binding:vnic_type"`
	// A dictionary that enables the application running on the specified
	// host to pass and receive virtual network interface (VIF) port-specific
	// information to the plug-in
	Profile map[string]string `json:"binding:profile"`
}

// ExtractPorts accepts a Page struct, specifically a PortPage struct,
// and extracts the elements into a slice of Port structs. In other words,
// a generic collection is mapped into a relevant slice.
func ExtractPorts(r pagination.Page) ([]Port, error) {
	var s struct {
		Ports []Port `json:"ports"`
	}
	err := (r.(ports.PortPage)).ExtractInto(&s)
	return s.Ports, err
}
