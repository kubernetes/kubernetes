package siteconnections

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// CreateOptsBuilder allows extensions to add additional parameters to the
// Create request.
type CreateOptsBuilder interface {
	ToConnectionCreateMap() (map[string]interface{}, error)
}
type Action string
type Initiator string

const (
	ActionHold             Action    = "hold"
	ActionClear            Action    = "clear"
	ActionRestart          Action    = "restart"
	ActionDisabled         Action    = "disabled"
	ActionRestartByPeer    Action    = "restart-by-peer"
	InitiatorBiDirectional Initiator = "bi-directional"
	InitiatorResponseOnly  Initiator = "response-only"
)

// DPDCreateOpts contains all the values needed to create a valid configuration for Dead Peer detection protocols
type DPDCreateOpts struct {
	// The dead peer detection (DPD) action.
	// A valid value is clear, hold, restart, disabled, or restart-by-peer.
	// Default value is hold.
	Action Action `json:"action,omitempty"`

	// The dead peer detection (DPD) timeout in seconds.
	// A valid value is a positive integer that is greater than the DPD interval value.
	// Default is 120.
	Timeout int `json:"timeout,omitempty"`

	// The dead peer detection (DPD) interval, in seconds.
	// A valid value is a positive integer.
	// Default is 30.
	Interval int `json:"interval,omitempty"`
}

// CreateOpts contains all the values needed to create a new IPSec site connection
type CreateOpts struct {
	// The ID of the IKE policy
	IKEPolicyID string `json:"ikepolicy_id"`

	// The ID of the VPN Service
	VPNServiceID string `json:"vpnservice_id"`

	// The ID for the endpoint group that contains private subnets for the local side of the connection.
	// You must specify this parameter with the peer_ep_group_id parameter unless
	// in backward- compatible mode where peer_cidrs is provided with a subnet_id for the VPN service.
	LocalEPGroupID string `json:"local_ep_group_id,omitempty"`

	// The ID of the IPsec policy.
	IPSecPolicyID string `json:"ipsecpolicy_id"`

	// The peer router identity for authentication.
	// A valid value is an IPv4 address, IPv6 address, e-mail address, key ID, or FQDN.
	// Typically, this value matches the peer_address value.
	PeerID string `json:"peer_id"`

	// The ID of the project
	TenantID string `json:"tenant_id,omitempty"`

	// The ID for the endpoint group that contains private CIDRs in the form < net_address > / < prefix >
	// for the peer side of the connection.
	// You must specify this parameter with the local_ep_group_id parameter unless in backward-compatible mode
	// where peer_cidrs is provided with a subnet_id for the VPN service.
	PeerEPGroupID string `json:"peer_ep_group_id,omitempty"`

	// An ID to be used instead of the external IP address for a virtual router used in traffic between instances on different networks in east-west traffic.
	// Most often, local ID would be domain name, email address, etc.
	// If this is not configured then the external IP address will be used as the ID.
	LocalID string `json:"local_id,omitempty"`

	// The human readable name of the connection.
	// Does not have to be unique.
	// Default is an empty string
	Name string `json:"name,omitempty"`

	// The human readable description of the connection.
	// Does not have to be unique.
	// Default is an empty string
	Description string `json:"description,omitempty"`

	// The peer gateway public IPv4 or IPv6 address or FQDN.
	PeerAddress string `json:"peer_address"`

	// The pre-shared key.
	// A valid value is any string.
	PSK string `json:"psk"`

	// Indicates whether this VPN can only respond to connections or both respond to and initiate connections.
	// A valid value is response-only or bi-directional. Default is bi-directional.
	Initiator Initiator `json:"initiator,omitempty"`

	// Unique list of valid peer private CIDRs in the form < net_address > / < prefix > .
	PeerCIDRs []string `json:"peer_cidrs,omitempty"`

	// The administrative state of the resource, which is up (true) or down (false).
	// Default is false
	AdminStateUp *bool `json:"admin_state_up,omitempty"`

	// A dictionary with dead peer detection (DPD) protocol controls.
	DPD *DPDCreateOpts `json:"dpd,omitempty"`

	// The maximum transmission unit (MTU) value to address fragmentation.
	// Minimum value is 68 for IPv4, and 1280 for IPv6.
	MTU int `json:"mtu,omitempty"`
}

// ToConnectionCreateMap casts a CreateOpts struct to a map.
func (opts CreateOpts) ToConnectionCreateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "ipsec_site_connection")
}

// Create accepts a CreateOpts struct and uses the values to create a new
// IPSec site connection.
func Create(c *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	b, err := opts.ToConnectionCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = c.Post(rootURL(c), b, &r.Body, nil)

	return
}

// Delete will permanently delete a particular IPSec site connection based on its
// unique ID.
func Delete(c *gophercloud.ServiceClient, id string) (r DeleteResult) {
	_, r.Err = c.Delete(resourceURL(c, id), nil)
	return
}

// Get retrieves a particular IPSec site connection based on its unique ID.
func Get(c *gophercloud.ServiceClient, id string) (r GetResult) {
	_, r.Err = c.Get(resourceURL(c, id), &r.Body, nil)
	return
}

// ListOptsBuilder allows extensions to add additional parameters to the
// List request.
type ListOptsBuilder interface {
	ToConnectionListQuery() (string, error)
}

// ListOpts allows the filtering and sorting of paginated collections through
// the API. Filtering is achieved by passing in struct field values that map to
// the IPSec site connection attributes you want to see returned.
type ListOpts struct {
	IKEPolicyID    string    `q:"ikepolicy_id"`
	VPNServiceID   string    `q:"vpnservice_id"`
	LocalEPGroupID string    `q:"local_ep_group_id"`
	IPSecPolicyID  string    `q:"ipsecpolicy_id"`
	PeerID         string    `q:"peer_id"`
	TenantID       string    `q:"tenant_id"`
	ProjectID      string    `q:"project_id"`
	PeerEPGroupID  string    `q:"peer_ep_group_id"`
	LocalID        string    `q:"local_id"`
	Name           string    `q:"name"`
	Description    string    `q:"description"`
	PeerAddress    string    `q:"peer_address"`
	PSK            string    `q:"psk"`
	Initiator      Initiator `q:"initiator"`
	AdminStateUp   *bool     `q:"admin_state_up"`
	MTU            int       `q:"mtu"`
}

// ToConnectionListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToConnectionListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// List returns a Pager which allows you to iterate over a collection of
// IPSec site connections. It accepts a ListOpts struct, which allows you to filter
// and sort the returned collection for greater efficiency.
func List(c *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := rootURL(c)
	if opts != nil {
		query, err := opts.ToConnectionListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}
	return pagination.NewPager(c, url, func(r pagination.PageResult) pagination.Page {
		return ConnectionPage{pagination.LinkedPageBase{PageResult: r}}
	})
}

// UpdateOptsBuilder allows extensions to add additional parameters to the
// Update request.
type UpdateOptsBuilder interface {
	ToConnectionUpdateMap() (map[string]interface{}, error)
}

// UpdateOpts contains the values used when updating the DPD of an IPSec site connection
type DPDUpdateOpts struct {
	Action   Action `json:"action,omitempty"`
	Timeout  int    `json:"timeout,omitempty"`
	Interval int    `json:"interval,omitempty"`
}

// UpdateOpts contains the values used when updating an IPSec site connection
type UpdateOpts struct {
	Description    *string        `json:"description,omitempty"`
	Name           *string        `json:"name,omitempty"`
	LocalID        string         `json:"local_id,omitempty"`
	PeerAddress    string         `json:"peer_address,omitempty"`
	PeerID         string         `json:"peer_id,omitempty"`
	PeerCIDRs      []string       `json:"peer_cidrs,omitempty"`
	LocalEPGroupID string         `json:"local_ep_group_id,omitempty"`
	PeerEPGroupID  string         `json:"peer_ep_group_id,omitempty"`
	MTU            int            `json:"mtu,omitempty"`
	Initiator      Initiator      `json:"initiator,omitempty"`
	PSK            string         `json:"psk,omitempty"`
	DPD            *DPDUpdateOpts `json:"dpd,omitempty"`
	AdminStateUp   *bool          `json:"admin_state_up,omitempty"`
}

// ToConnectionUpdateMap casts an UpdateOpts struct to a map.
func (opts UpdateOpts) ToConnectionUpdateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "ipsec_site_connection")
}

// Update allows IPSec site connections to be updated.
func Update(c *gophercloud.ServiceClient, id string, opts UpdateOptsBuilder) (r UpdateResult) {
	b, err := opts.ToConnectionUpdateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = c.Put(resourceURL(c, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}
