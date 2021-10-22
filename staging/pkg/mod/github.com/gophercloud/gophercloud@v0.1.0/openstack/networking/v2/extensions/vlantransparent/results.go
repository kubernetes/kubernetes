package vlantransparent

// TransparentExt represents a decorated form of a network with
// "vlan-transparent" extension attributes.
type TransparentExt struct {
	// VLANTransparent whether the network is a VLAN transparent network or not.
	VLANTransparent bool `json:"vlan_transparent"`
}
