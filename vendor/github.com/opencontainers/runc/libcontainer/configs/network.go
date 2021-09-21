package configs

// Network defines configuration for a container's networking stack
//
// The network configuration can be omitted from a container causing the
// container to be setup with the host's networking stack
type Network struct {
	// Type sets the networks type, commonly veth and loopback
	Type string `json:"type"`

	// Name of the network interface
	Name string `json:"name"`

	// The bridge to use.
	Bridge string `json:"bridge"`

	// MacAddress contains the MAC address to set on the network interface
	MacAddress string `json:"mac_address"`

	// Address contains the IPv4 and mask to set on the network interface
	Address string `json:"address"`

	// Gateway sets the gateway address that is used as the default for the interface
	Gateway string `json:"gateway"`

	// IPv6Address contains the IPv6 and mask to set on the network interface
	IPv6Address string `json:"ipv6_address"`

	// IPv6Gateway sets the ipv6 gateway address that is used as the default for the interface
	IPv6Gateway string `json:"ipv6_gateway"`

	// Mtu sets the mtu value for the interface and will be mirrored on both the host and
	// container's interfaces if a pair is created, specifically in the case of type veth
	// Note: This does not apply to loopback interfaces.
	Mtu int `json:"mtu"`

	// TxQueueLen sets the tx_queuelen value for the interface and will be mirrored on both the host and
	// container's interfaces if a pair is created, specifically in the case of type veth
	// Note: This does not apply to loopback interfaces.
	TxQueueLen int `json:"txqueuelen"`

	// HostInterfaceName is a unique name of a veth pair that resides on in the host interface of the
	// container.
	HostInterfaceName string `json:"host_interface_name"`

	// HairpinMode specifies if hairpin NAT should be enabled on the virtual interface
	// bridge port in the case of type veth
	// Note: This is unsupported on some systems.
	// Note: This does not apply to loopback interfaces.
	HairpinMode bool `json:"hairpin_mode"`
}

// Route defines a routing table entry.
//
// Routes can be specified to create entries in the routing table as the container
// is started.
//
// All of destination, source, and gateway should be either IPv4 or IPv6.
// One of the three options must be present, and omitted entries will use their
// IP family default for the route table.  For IPv4 for example, setting the
// gateway to 1.2.3.4 and the interface to eth0 will set up a standard
// destination of 0.0.0.0(or *) when viewed in the route table.
type Route struct {
	// Destination specifies the destination IP address and mask in the CIDR form.
	Destination string `json:"destination"`

	// Source specifies the source IP address and mask in the CIDR form.
	Source string `json:"source"`

	// Gateway specifies the gateway IP address.
	Gateway string `json:"gateway"`

	// InterfaceName specifies the device to set this route up for, for example eth0.
	InterfaceName string `json:"interface_name"`
}
