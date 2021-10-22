package siteconnections

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

type DPD struct {
	// Action is the dead peer detection (DPD) action.
	Action string `json:"action"`

	// Timeout is the dead peer detection (DPD) timeout in seconds.
	Timeout int `json:"timeout"`

	// Interval is the dead peer detection (DPD) interval in seconds.
	Interval int `json:"interval"`
}

// Connection is an IPSec site connection
type Connection struct {
	// IKEPolicyID is the ID of the IKE policy.
	IKEPolicyID string `json:"ikepolicy_id"`

	// VPNServiceID is the ID of the VPN service.
	VPNServiceID string `json:"vpnservice_id"`

	// LocalEPGroupID is the ID for the endpoint group that contains private subnets for the local side of the connection.
	LocalEPGroupID string `json:"local_ep_group_id"`

	// IPSecPolicyID is the ID of the IPSec policy
	IPSecPolicyID string `json:"ipsecpolicy_id"`

	// PeerID is the peer router identity for authentication.
	PeerID string `json:"peer_id"`

	// TenantID is the ID of the project.
	TenantID string `json:"tenant_id"`

	// ProjectID is the ID of the project.
	ProjectID string `json:"project_id"`

	// PeerEPGroupID is the ID for the endpoint group that contains private CIDRs in the form < net_address > / < prefix >
	// for the peer side of the connection.
	PeerEPGroupID string `json:"peer_ep_group_id"`

	// LocalID is an ID to be used instead of the external IP address for a virtual router used in traffic
	// between instances on different networks in east-west traffic.
	LocalID string `json:"local_id"`

	// Name is the human readable name of the connection.
	Name string `json:"name"`

	// Description is the human readable description of the connection.
	Description string `json:"description"`

	// PeerAddress is the peer gateway public IPv4 or IPv6 address or FQDN.
	PeerAddress string `json:"peer_address"`

	// RouteMode is the route mode.
	RouteMode string `json:"route_mode"`

	// PSK is the pre-shared key.
	PSK string `json:"psk"`

	// Initiator indicates whether this VPN can only respond to connections or both respond to and initiate connections.
	Initiator string `json:"initiator"`

	// PeerCIDRs is a unique list of valid peer private CIDRs in the form < net_address > / < prefix > .
	PeerCIDRs []string `json:"peer_cidrs"`

	// AdminStateUp is the administrative state of the connection.
	AdminStateUp bool `json:"admin_state_up"`

	// DPD is the dead peer detection (DPD) protocol controls.
	DPD DPD `json:"dpd"`

	// AuthMode is the authentication mode.
	AuthMode string `json:"auth_mode"`

	// MTU is the maximum transmission unit (MTU) value to address fragmentation.
	MTU int `json:"mtu"`

	// Status indicates whether the IPsec connection is currently operational.
	// Values are ACTIVE, DOWN, BUILD, ERROR, PENDING_CREATE, PENDING_UPDATE, or PENDING_DELETE.
	Status string `json:"status"`

	// ID is the id of the connection
	ID string `json:"id"`
}

type commonResult struct {
	gophercloud.Result
}

// ConnectionPage is the page returned by a pager when traversing over a
// collection of IPSec site connections.
type ConnectionPage struct {
	pagination.LinkedPageBase
}

// NextPageURL is invoked when a paginated collection of IPSec site connections has
// reached the end of a page and the pager seeks to traverse over a new one.
// In order to do this, it needs to construct the next page's URL.
func (r ConnectionPage) NextPageURL() (string, error) {
	var s struct {
		Links []gophercloud.Link `json:"ipsec_site_connections_links"`
	}
	err := r.ExtractInto(&s)
	if err != nil {
		return "", err
	}
	return gophercloud.ExtractNextURL(s.Links)
}

// IsEmpty checks whether a ConnectionPage struct is empty.
func (r ConnectionPage) IsEmpty() (bool, error) {
	is, err := ExtractConnections(r)
	return len(is) == 0, err
}

// ExtractConnections accepts a Page struct, specifically a Connection struct,
// and extracts the elements into a slice of Connection structs. In other words,
// a generic collection is mapped into a relevant slice.
func ExtractConnections(r pagination.Page) ([]Connection, error) {
	var s struct {
		Connections []Connection `json:"ipsec_site_connections"`
	}
	err := (r.(ConnectionPage)).ExtractInto(&s)
	return s.Connections, err
}

// Extract is a function that accepts a result and extracts an IPSec site connection.
func (r commonResult) Extract() (*Connection, error) {
	var s struct {
		Connection *Connection `json:"ipsec_site_connection"`
	}
	err := r.ExtractInto(&s)
	return s.Connection, err
}

// CreateResult represents the result of a create operation. Call its Extract
// method to interpret it as a Connection.
type CreateResult struct {
	commonResult
}

// DeleteResult represents the result of a delete operation. Call its
// ExtractErr method to determine if the operation succeeded or failed.
type DeleteResult struct {
	gophercloud.ErrResult
}

// GetResult represents the result of a get operation. Call its Extract
// method to interpret it as a Connection.
type GetResult struct {
	commonResult
}

// UpdateResult represents the result of an update operation. Call its Extract
// method to interpret it as a connection
type UpdateResult struct {
	commonResult
}
