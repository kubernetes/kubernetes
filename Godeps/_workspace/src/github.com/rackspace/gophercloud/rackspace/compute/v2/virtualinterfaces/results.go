package virtualinterfaces

import (
	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

type commonResult struct {
	gophercloud.Result
}

// Extract is a function that accepts a result and extracts a network resource.
func (r commonResult) Extract() (*VirtualInterface, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var res struct {
		VirtualInterfaces []VirtualInterface `mapstructure:"virtual_interfaces" json:"virtual_interfaces"`
	}

	err := mapstructure.Decode(r.Body, &res)

	return &res.VirtualInterfaces[0], err
}

// CreateResult represents the result of a create operation.
type CreateResult struct {
	commonResult
}

// DeleteResult represents the result of a delete operation.
type DeleteResult struct {
	gophercloud.ErrResult
}

// IPAddress represents a vitual address attached to a VirtualInterface.
type IPAddress struct {
	Address      string `mapstructure:"address" json:"address"`
	NetworkID    string `mapstructure:"network_id" json:"network_id"`
	NetworkLabel string `mapstructure:"network_label" json:"network_label"`
}

// VirtualInterface represents a virtual interface.
type VirtualInterface struct {
	// UUID for the virtual interface
	ID string `mapstructure:"id" json:"id"`

	MACAddress string `mapstructure:"mac_address" json:"mac_address"`

	IPAddresses []IPAddress `mapstructure:"ip_addresses" json:"ip_addresses"`
}

// VirtualInterfacePage is the page returned by a pager when traversing over a
// collection of virtual interfaces.
type VirtualInterfacePage struct {
	pagination.SinglePageBase
}

// IsEmpty returns true if the NetworkPage contains no Networks.
func (r VirtualInterfacePage) IsEmpty() (bool, error) {
	networks, err := ExtractVirtualInterfaces(r)
	if err != nil {
		return true, err
	}
	return len(networks) == 0, nil
}

// ExtractVirtualInterfaces accepts a Page struct, specifically a VirtualInterfacePage struct,
// and extracts the elements into a slice of VirtualInterface structs. In other words,
// a generic collection is mapped into a relevant slice.
func ExtractVirtualInterfaces(page pagination.Page) ([]VirtualInterface, error) {
	var resp struct {
		VirtualInterfaces []VirtualInterface `mapstructure:"virtual_interfaces" json:"virtual_interfaces"`
	}

	err := mapstructure.Decode(page.(VirtualInterfacePage).Body, &resp)

	return resp.VirtualInterfaces, err
}
