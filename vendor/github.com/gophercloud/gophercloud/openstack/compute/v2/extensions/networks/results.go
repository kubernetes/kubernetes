package networks

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// A Network represents a nova-network that an instance communicates on
type Network struct {
	// The Bridge that VIFs on this network are connected to
	Bridge string `json:"bridge"`

	// BridgeInterface is what interface is connected to the Bridge
	BridgeInterface string `json:"bridge_interface"`

	// The Broadcast address of the network.
	Broadcast string `json:"broadcast"`

	// CIDR is the IPv4 subnet.
	CIDR string `json:"cidr"`

	// CIDRv6 is the IPv6 subnet.
	CIDRv6 string `json:"cidr_v6"`

	// CreatedAt is when the network was created..
	CreatedAt gophercloud.JSONRFC3339MilliNoZ `json:"created_at,omitempty"`

	// Deleted shows if the network has been deleted.
	Deleted bool `json:"deleted"`

	// DeletedAt is the time when the network was deleted.
	DeletedAt gophercloud.JSONRFC3339MilliNoZ `json:"deleted_at,omitempty"`

	// DHCPStart is the start of the DHCP address range.
	DHCPStart string `json:"dhcp_start"`

	// DNS1 is the first DNS server to use through DHCP.
	DNS1 string `json:"dns_1"`

	// DNS2 is the first DNS server to use through DHCP.
	DNS2 string `json:"dns_2"`

	// Gateway is the network gateway.
	Gateway string `json:"gateway"`

	// Gatewayv6 is the IPv6 network gateway.
	Gatewayv6 string `json:"gateway_v6"`

	// Host is the host that the network service is running on.
	Host string `json:"host"`

	// ID is the UUID of the network.
	ID string `json:"id"`

	// Injected determines if network information is injected into the host.
	Injected bool `json:"injected"`

	// Label is the common name that the network has..
	Label string `json:"label"`

	// MultiHost is if multi-host networking is enablec..
	MultiHost bool `json:"multi_host"`

	// Netmask is the network netmask.
	Netmask string `json:"netmask"`

	// Netmaskv6 is the IPv6 netmask.
	Netmaskv6 string `json:"netmask_v6"`

	// Priority is the network interface priority.
	Priority int `json:"priority"`

	// ProjectID is the project associated with this network.
	ProjectID string `json:"project_id"`

	// RXTXBase configures bandwidth entitlement.
	RXTXBase int `json:"rxtx_base"`

	// UpdatedAt is the time when the network was last updated.
	UpdatedAt gophercloud.JSONRFC3339MilliNoZ `json:"updated_at,omitempty"`

	// VLAN is the vlan this network runs on.
	VLAN int `json:"vlan"`

	// VPNPrivateAddress is the private address of the CloudPipe VPN.
	VPNPrivateAddress string `json:"vpn_private_address"`

	// VPNPublicAddress is the public address of the CloudPipe VPN.
	VPNPublicAddress string `json:"vpn_public_address"`

	// VPNPublicPort is the port of the CloudPipe VPN.
	VPNPublicPort int `json:"vpn_public_port"`
}

// NetworkPage stores a single, only page of Networks
// results from a List call.
type NetworkPage struct {
	pagination.SinglePageBase
}

// IsEmpty determines whether or not a NetworkPage is empty.
func (page NetworkPage) IsEmpty() (bool, error) {
	va, err := ExtractNetworks(page)
	return len(va) == 0, err
}

// ExtractNetworks interprets a page of results as a slice of Networks
func ExtractNetworks(r pagination.Page) ([]Network, error) {
	var s struct {
		Networks []Network `json:"networks"`
	}
	err := (r.(NetworkPage)).ExtractInto(&s)
	return s.Networks, err
}

type NetworkResult struct {
	gophercloud.Result
}

// Extract is a method that attempts to interpret any Network resource
// response as a Network struct.
func (r NetworkResult) Extract() (*Network, error) {
	var s struct {
		Network *Network `json:"network"`
	}
	err := r.ExtractInto(&s)
	return s.Network, err
}

// GetResult is the response from a Get operation. Call its Extract method to interpret it
// as a Network.
type GetResult struct {
	NetworkResult
}
