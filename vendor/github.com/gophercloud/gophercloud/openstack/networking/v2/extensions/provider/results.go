package provider

import (
	"encoding/json"
	"strconv"

	"github.com/gophercloud/gophercloud/openstack/networking/v2/networks"
	"github.com/gophercloud/gophercloud/pagination"
)

// NetworkExtAttrs represents an extended form of a Network with additional fields.
type NetworkExtAttrs struct {
	// UUID for the network
	ID string `json:"id"`

	// Human-readable name for the network. Might not be unique.
	Name string `json:"name"`

	// The administrative state of network. If false (down), the network does not forward packets.
	AdminStateUp bool `json:"admin_state_up"`

	// Indicates whether network is currently operational. Possible values include
	// `ACTIVE', `DOWN', `BUILD', or `ERROR'. Plug-ins might define additional values.
	Status string `json:"status"`

	// Subnets associated with this network.
	Subnets []string `json:"subnets"`

	// Owner of network. Only admin users can specify a tenant_id other than its own.
	TenantID string `json:"tenant_id"`

	// Specifies whether the network resource can be accessed by any tenant or not.
	Shared bool `json:"shared"`

	// Specifies the nature of the physical network mapped to this network
	// resource. Examples are flat, vlan, or gre.
	NetworkType string `json:"provider:network_type"`

	// Identifies the physical network on top of which this network object is
	// being implemented. The OpenStack Networking API does not expose any facility
	// for retrieving the list of available physical networks. As an example, in
	// the Open vSwitch plug-in this is a symbolic name which is then mapped to
	// specific bridges on each compute host through the Open vSwitch plug-in
	// configuration file.
	PhysicalNetwork string `json:"provider:physical_network"`

	// Identifies an isolated segment on the physical network; the nature of the
	// segment depends on the segmentation model defined by network_type. For
	// instance, if network_type is vlan, then this is a vlan identifier;
	// otherwise, if network_type is gre, then this will be a gre key.
	SegmentationID string `json:"provider:segmentation_id"`
}

func (n *NetworkExtAttrs) UnmarshalJSON(b []byte) error {
	type tmp NetworkExtAttrs
	var networkExtAttrs *struct {
		tmp
		SegmentationID interface{} `json:"provider:segmentation_id"`
	}

	if err := json.Unmarshal(b, &networkExtAttrs); err != nil {
		return err
	}

	*n = NetworkExtAttrs(networkExtAttrs.tmp)

	switch t := networkExtAttrs.SegmentationID.(type) {
	case float64:
		n.SegmentationID = strconv.FormatFloat(t, 'f', -1, 64)
	case string:
		n.SegmentationID = string(t)
	}

	return nil
}

// ExtractGet decorates a GetResult struct returned from a networks.Get()
// function with extended attributes.
func ExtractGet(r networks.GetResult) (*NetworkExtAttrs, error) {
	var s struct {
		Network *NetworkExtAttrs `json:"network"`
	}
	err := r.ExtractInto(&s)
	return s.Network, err
}

// ExtractCreate decorates a CreateResult struct returned from a networks.Create()
// function with extended attributes.
func ExtractCreate(r networks.CreateResult) (*NetworkExtAttrs, error) {
	var s struct {
		Network *NetworkExtAttrs `json:"network"`
	}
	err := r.ExtractInto(&s)
	return s.Network, err
}

// ExtractUpdate decorates a UpdateResult struct returned from a
// networks.Update() function with extended attributes.
func ExtractUpdate(r networks.UpdateResult) (*NetworkExtAttrs, error) {
	var s struct {
		Network *NetworkExtAttrs `json:"network"`
	}
	err := r.ExtractInto(&s)
	return s.Network, err
}

// ExtractList accepts a Page struct, specifically a NetworkPage struct, and
// extracts the elements into a slice of NetworkExtAttrs structs. In other
// words, a generic collection is mapped into a relevant slice.
func ExtractList(r pagination.Page) ([]NetworkExtAttrs, error) {
	var s struct {
		Networks []NetworkExtAttrs `json:"networks" json:"networks"`
	}
	err := (r.(networks.NetworkPage)).ExtractInto(&s)
	return s.Networks, err
}
