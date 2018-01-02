package portsbinding

// IP is a sub-struct that represents an individual IP.
type IP struct {
	SubnetID  string `json:"subnet_id"`
	IPAddress string `json:"ip_address"`
}

// PortsBindingExt represents a decorated form of a Port with the additional
// port binding information.
type PortsBindingExt struct {
	// The ID of the host where the port is allocated.
	HostID string `json:"binding:host_id"`

	// A dictionary that enables the application to pass information about
	// functions that the Networking API provides.
	VIFDetails map[string]interface{} `json:"binding:vif_details"`

	// The VIF type for the port.
	VIFType string `json:"binding:vif_type"`

	// The virtual network interface card (vNIC) type that is bound to the
	// neutron port.
	VNICType string `json:"binding:vnic_type"`

	// A dictionary that enables the application running on the specified
	// host to pass and receive virtual network interface (VIF) port-specific
	// information to the plug-in.
	Profile map[string]string `json:"binding:profile"`
}
