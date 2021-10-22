package provider

import (
	"encoding/json"
	"strconv"
)

// NetworkProviderExt represents an extended form of a Network with additional
// fields.
type NetworkProviderExt struct {
	// Specifies the nature of the physical network mapped to this network
	// resource. Examples are flat, vlan, or gre.
	NetworkType string `json:"provider:network_type"`

	// Identifies the physical network on top of which this network object is
	// being implemented. The OpenStack Networking API does not expose any
	// facility for retrieving the list of available physical networks. As an
	// example, in the Open vSwitch plug-in this is a symbolic name which is
	// then mapped to specific bridges on each compute host through the Open
	// vSwitch plug-in configuration file.
	PhysicalNetwork string `json:"provider:physical_network"`

	// Identifies an isolated segment on the physical network; the nature of the
	// segment depends on the segmentation model defined by network_type. For
	// instance, if network_type is vlan, then this is a vlan identifier;
	// otherwise, if network_type is gre, then this will be a gre key.
	SegmentationID string `json:"-"`

	// Segments is an array of Segment which defines multiple physical bindings
	// to logical networks.
	Segments []Segment `json:"segments"`
}

// Segment defines a physical binding to a logical network.
type Segment struct {
	PhysicalNetwork string `json:"provider:physical_network"`
	NetworkType     string `json:"provider:network_type"`
	SegmentationID  int    `json:"provider:segmentation_id"`
}

func (r *NetworkProviderExt) UnmarshalJSON(b []byte) error {
	type tmp NetworkProviderExt
	var networkProviderExt struct {
		tmp
		SegmentationID interface{} `json:"provider:segmentation_id"`
	}

	if err := json.Unmarshal(b, &networkProviderExt); err != nil {
		return err
	}

	*r = NetworkProviderExt(networkProviderExt.tmp)

	switch t := networkProviderExt.SegmentationID.(type) {
	case float64:
		r.SegmentationID = strconv.FormatFloat(t, 'f', -1, 64)
	case string:
		r.SegmentationID = string(t)
	}

	return nil
}
