package ports

import (
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

type portResult struct {
	gophercloud.Result
}

func (r portResult) Extract() (*Port, error) {
	var s Port
	err := r.ExtractInto(&s)
	return &s, err
}

func (r portResult) ExtractInto(v interface{}) error {
	return r.Result.ExtractIntoStructPtr(v, "")
}

func ExtractPortsInto(r pagination.Page, v interface{}) error {
	return r.(PortPage).Result.ExtractIntoSlicePtr(v, "ports")
}

// Port represents a port in the OpenStack Bare Metal API.
type Port struct {
	// UUID for the resource.
	UUID string `json:"uuid"`

	// Physical hardware address of this network Port,
	// typically the hardware MAC address.
	Address string `json:"address"`

	// UUID of the Node this resource belongs to.
	NodeUUID string `json:"node_uuid"`

	// UUID of the Portgroup this resource belongs to.
	PortGroupUUID string `json:"portgroup_uuid"`

	// The Port binding profile. If specified, must contain switch_id (only a MAC
	// address or an OpenFlow based datapath_id of the switch are accepted in this
	// field) and port_id (identifier of the physical port on the switch to which
	// node’s port is connected to) fields. switch_info is an optional string
	// field to be used to store any vendor-specific information.
	LocalLinkConnection map[string]interface{} `json:"local_link_connection"`

	// Indicates whether PXE is enabled or disabled on the Port.
	PXEEnabled bool `json:"pxe_enabled"`

	// The name of the physical network to which a port is connected.
	// May be empty.
	PhysicalNetwork string `json:"physical_network"`

	// Internal metadata set and stored by the Port. This field is read-only.
	InternalInfo map[string]interface{} `json:"internal_info"`

	// A set of one or more arbitrary metadata key and value pairs.
	Extra map[string]interface{} `json:"extra"`

	// The UTC date and time when the resource was created, ISO 8601 format.
	CreatedAt time.Time `json:"created_at"`

	// The UTC date and time when the resource was updated, ISO 8601 format.
	// May be “null”.
	UpdatedAt time.Time `json:"updated_at"`

	// A list of relative links. Includes the self and bookmark links.
	Links []interface{} `json:"links"`

	// Indicates whether the Port is a Smart NIC port.
	IsSmartNIC bool `json:"is_smartnic"`
}

// PortPage abstracts the raw results of making a List() request against
// the API.
type PortPage struct {
	pagination.LinkedPageBase
}

// IsEmpty returns true if a page contains no Port results.
func (r PortPage) IsEmpty() (bool, error) {
	s, err := ExtractPorts(r)
	return len(s) == 0, err
}

// NextPageURL uses the response's embedded link reference to navigate to the
// next page of results.
func (r PortPage) NextPageURL() (string, error) {
	var s struct {
		Links []gophercloud.Link `json:"ports_links"`
	}
	err := r.ExtractInto(&s)
	if err != nil {
		return "", err
	}
	return gophercloud.ExtractNextURL(s.Links)
}

// ExtractPorts interprets the results of a single page from a List() call,
// producing a slice of Port entities.
func ExtractPorts(r pagination.Page) ([]Port, error) {
	var s []Port
	err := ExtractPortsInto(r, &s)
	return s, err
}

// GetResult is the response from a Get operation. Call its Extract
// method to interpret it as a Port.
type GetResult struct {
	portResult
}

// CreateResult is the response from a Create operation.
type CreateResult struct {
	portResult
}

// UpdateResult is the response from an Update operation. Call its Extract
// method to interpret it as a Port.
type UpdateResult struct {
	portResult
}

// DeleteResult is the response from a Delete operation. Call its ExtractErr
// method to determine if the call succeeded or failed.
type DeleteResult struct {
	gophercloud.ErrResult
}
