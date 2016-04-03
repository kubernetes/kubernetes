package networks

import (
	"fmt"
	"time"

	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// A Network represents a nova-network that an instance communicates on
type Network struct {
	// The Bridge that VIFs on this network are connected to
	Bridge string `mapstructure:"bridge"`

	// BridgeInterface is what interface is connected to the Bridge
	BridgeInterface string `mapstructure:"bridge_interface"`

	// The Broadcast address of the network.
	Broadcast string `mapstructure:"broadcast"`

	// CIDR is the IPv4 subnet.
	CIDR string `mapstructure:"cidr"`

	// CIDRv6 is the IPv6 subnet.
	CIDRv6 string `mapstructure:"cidr_v6"`

	// CreatedAt is when the network was created..
	CreatedAt time.Time `mapstructure:"-"`

	// Deleted shows if the network has been deleted.
	Deleted bool `mapstructure:"deleted"`

	// DeletedAt is the time when the network was deleted.
	DeletedAt time.Time `mapstructure:"-"`

	// DHCPStart is the start of the DHCP address range.
	DHCPStart string `mapstructure:"dhcp_start"`

	// DNS1 is the first DNS server to use through DHCP.
	DNS1 string `mapstructure:"dns_1"`

	// DNS2 is the first DNS server to use through DHCP.
	DNS2 string `mapstructure:"dns_2"`

	// Gateway is the network gateway.
	Gateway string `mapstructure:"gateway"`

	// Gatewayv6 is the IPv6 network gateway.
	Gatewayv6 string `mapstructure:"gateway_v6"`

	// Host is the host that the network service is running on.
	Host string `mapstructure:"host"`

	// ID is the UUID of the network.
	ID string `mapstructure:"id"`

	// Injected determines if network information is injected into the host.
	Injected bool `mapstructure:"injected"`

	// Label is the common name that the network has..
	Label string `mapstructure:"label"`

	// MultiHost is if multi-host networking is enablec..
	MultiHost bool `mapstructure:"multi_host"`

	// Netmask is the network netmask.
	Netmask string `mapstructure:"netmask"`

	// Netmaskv6 is the IPv6 netmask.
	Netmaskv6 string `mapstructure:"netmask_v6"`

	// Priority is the network interface priority.
	Priority int `mapstructure:"priority"`

	// ProjectID is the project associated with this network.
	ProjectID string `mapstructure:"project_id"`

	// RXTXBase configures bandwidth entitlement.
	RXTXBase int `mapstructure:"rxtx_base"`

	// UpdatedAt is the time when the network was last updated.
	UpdatedAt time.Time `mapstructure:"-"`

	// VLAN is the vlan this network runs on.
	VLAN int `mapstructure:"vlan"`

	// VPNPrivateAddress is the private address of the CloudPipe VPN.
	VPNPrivateAddress string `mapstructure:"vpn_private_address"`

	// VPNPublicAddress is the public address of the CloudPipe VPN.
	VPNPublicAddress string `mapstructure:"vpn_public_address"`

	// VPNPublicPort is the port of the CloudPipe VPN.
	VPNPublicPort int `mapstructure:"vpn_public_port"`
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
func ExtractNetworks(page pagination.Page) ([]Network, error) {
	var res struct {
		Networks []Network `mapstructure:"networks"`
	}

	err := mapstructure.Decode(page.(NetworkPage).Body, &res)

	var rawNetworks []interface{}
	body := page.(NetworkPage).Body
	switch body.(type) {
	case map[string]interface{}:
		rawNetworks = body.(map[string]interface{})["networks"].([]interface{})
	case map[string][]interface{}:
		rawNetworks = body.(map[string][]interface{})["networks"]
	default:
		return res.Networks, fmt.Errorf("Unknown type")
	}

	for i := range rawNetworks {
		thisNetwork := rawNetworks[i].(map[string]interface{})
		if t, ok := thisNetwork["created_at"].(string); ok && t != "" {
			createdAt, err := time.Parse("2006-01-02 15:04:05.000000", t)
			if err != nil {
				return res.Networks, err
			}
			res.Networks[i].CreatedAt = createdAt
		}

		if t, ok := thisNetwork["updated_at"].(string); ok && t != "" {
			updatedAt, err := time.Parse("2006-01-02 15:04:05.000000", t)
			if err != nil {
				return res.Networks, err
			}
			res.Networks[i].UpdatedAt = updatedAt
		}

		if t, ok := thisNetwork["deleted_at"].(string); ok && t != "" {
			deletedAt, err := time.Parse("2006-01-02 15:04:05.000000", t)
			if err != nil {
				return res.Networks, err
			}
			res.Networks[i].DeletedAt = deletedAt
		}
	}

	return res.Networks, err
}

type NetworkResult struct {
	gophercloud.Result
}

// Extract is a method that attempts to interpret any Network resource
// response as a Network struct.
func (r NetworkResult) Extract() (*Network, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var res struct {
		Network *Network `json:"network" mapstructure:"network"`
	}

	config := &mapstructure.DecoderConfig{
		Result:           &res,
		WeaklyTypedInput: true,
	}
	decoder, err := mapstructure.NewDecoder(config)
	if err != nil {
		return nil, err
	}

	if err := decoder.Decode(r.Body); err != nil {
		return nil, err
	}

	b := r.Body.(map[string]interface{})["network"].(map[string]interface{})

	if t, ok := b["created_at"].(string); ok && t != "" {
		createdAt, err := time.Parse("2006-01-02 15:04:05.000000", t)
		if err != nil {
			return res.Network, err
		}
		res.Network.CreatedAt = createdAt
	}

	if t, ok := b["updated_at"].(string); ok && t != "" {
		updatedAt, err := time.Parse("2006-01-02 15:04:05.000000", t)
		if err != nil {
			return res.Network, err
		}
		res.Network.UpdatedAt = updatedAt
	}

	if t, ok := b["deleted_at"].(string); ok && t != "" {
		deletedAt, err := time.Parse("2006-01-02 15:04:05.000000", t)
		if err != nil {
			return res.Network, err
		}
		res.Network.DeletedAt = deletedAt
	}

	return res.Network, err

}

// GetResult is the response from a Get operation. Call its Extract method to interpret it
// as a Network.
type GetResult struct {
	NetworkResult
}
