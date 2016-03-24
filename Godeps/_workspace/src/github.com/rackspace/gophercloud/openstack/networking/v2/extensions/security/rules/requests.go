package rules

import (
	"fmt"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// ListOpts allows the filtering and sorting of paginated collections through
// the API. Filtering is achieved by passing in struct field values that map to
// the security group attributes you want to see returned. SortKey allows you to
// sort by a particular network attribute. SortDir sets the direction, and is
// either `asc' or `desc'. Marker and Limit are used for pagination.
type ListOpts struct {
	Direction      string `q:"direction"`
	EtherType      string `q:"ethertype"`
	ID             string `q:"id"`
	PortRangeMax   int    `q:"port_range_max"`
	PortRangeMin   int    `q:"port_range_min"`
	Protocol       string `q:"protocol"`
	RemoteGroupID  string `q:"remote_group_id"`
	RemoteIPPrefix string `q:"remote_ip_prefix"`
	SecGroupID     string `q:"security_group_id"`
	TenantID       string `q:"tenant_id"`
	Limit          int    `q:"limit"`
	Marker         string `q:"marker"`
	SortKey        string `q:"sort_key"`
	SortDir        string `q:"sort_dir"`
}

// List returns a Pager which allows you to iterate over a collection of
// security group rules. It accepts a ListOpts struct, which allows you to filter
// and sort the returned collection for greater efficiency.
func List(c *gophercloud.ServiceClient, opts ListOpts) pagination.Pager {
	q, err := gophercloud.BuildQueryString(&opts)
	if err != nil {
		return pagination.Pager{Err: err}
	}
	u := rootURL(c) + q.String()
	return pagination.NewPager(c, u, func(r pagination.PageResult) pagination.Page {
		return SecGroupRulePage{pagination.LinkedPageBase{PageResult: r}}
	})
}

// Errors
var (
	errValidDirectionRequired = fmt.Errorf("A valid Direction is required")
	errValidEtherTypeRequired = fmt.Errorf("A valid EtherType is required")
	errSecGroupIDRequired     = fmt.Errorf("A valid SecGroupID is required")
	errValidProtocolRequired  = fmt.Errorf("A valid Protocol is required")
)

// Constants useful for CreateOpts
const (
	DirIngress   = "ingress"
	DirEgress    = "egress"
	Ether4       = "IPv4"
	Ether6       = "IPv6"
	ProtocolTCP  = "tcp"
	ProtocolUDP  = "udp"
	ProtocolICMP = "icmp"
)

// CreateOpts contains all the values needed to create a new security group rule.
type CreateOpts struct {
	// Required. Must be either "ingress" or "egress": the direction in which the
	// security group rule is applied.
	Direction string

	// Required. Must be "IPv4" or "IPv6", and addresses represented in CIDR must
	// match the ingress or egress rules.
	EtherType string

	// Required. The security group ID to associate with this security group rule.
	SecGroupID string

	// Optional. The maximum port number in the range that is matched by the
	// security group rule. The PortRangeMin attribute constrains the PortRangeMax
	// attribute. If the protocol is ICMP, this value must be an ICMP type.
	PortRangeMax int

	// Optional. The minimum port number in the range that is matched by the
	// security group rule. If the protocol is TCP or UDP, this value must be
	// less than or equal to the value of the PortRangeMax attribute. If the
	// protocol is ICMP, this value must be an ICMP type.
	PortRangeMin int

	// Optional. The protocol that is matched by the security group rule. Valid
	// values are "tcp", "udp", "icmp" or an empty string.
	Protocol string

	// Optional. The remote group ID to be associated with this security group
	// rule. You can specify either RemoteGroupID or RemoteIPPrefix.
	RemoteGroupID string

	// Optional. The remote IP prefix to be associated with this security group
	// rule. You can specify either RemoteGroupID or RemoteIPPrefix. This
	// attribute matches the specified IP prefix as the source IP address of the
	// IP packet.
	RemoteIPPrefix string

	// Required for admins. Indicates the owner of the VIP.
	TenantID string
}

// Create is an operation which adds a new security group rule and associates it
// with an existing security group (whose ID is specified in CreateOpts).
func Create(c *gophercloud.ServiceClient, opts CreateOpts) CreateResult {
	var res CreateResult

	// Validate required opts
	if opts.Direction != DirIngress && opts.Direction != DirEgress {
		res.Err = errValidDirectionRequired
		return res
	}
	if opts.EtherType != Ether4 && opts.EtherType != Ether6 {
		res.Err = errValidEtherTypeRequired
		return res
	}
	if opts.SecGroupID == "" {
		res.Err = errSecGroupIDRequired
		return res
	}
	if opts.Protocol != "" && opts.Protocol != ProtocolTCP && opts.Protocol != ProtocolUDP && opts.Protocol != ProtocolICMP {
		res.Err = errValidProtocolRequired
		return res
	}

	type secrule struct {
		Direction      string `json:"direction"`
		EtherType      string `json:"ethertype"`
		SecGroupID     string `json:"security_group_id"`
		PortRangeMax   int    `json:"port_range_max,omitempty"`
		PortRangeMin   int    `json:"port_range_min,omitempty"`
		Protocol       string `json:"protocol,omitempty"`
		RemoteGroupID  string `json:"remote_group_id,omitempty"`
		RemoteIPPrefix string `json:"remote_ip_prefix,omitempty"`
		TenantID       string `json:"tenant_id,omitempty"`
	}

	type request struct {
		SecRule secrule `json:"security_group_rule"`
	}

	reqBody := request{SecRule: secrule{
		Direction:      opts.Direction,
		EtherType:      opts.EtherType,
		SecGroupID:     opts.SecGroupID,
		PortRangeMax:   opts.PortRangeMax,
		PortRangeMin:   opts.PortRangeMin,
		Protocol:       opts.Protocol,
		RemoteGroupID:  opts.RemoteGroupID,
		RemoteIPPrefix: opts.RemoteIPPrefix,
		TenantID:       opts.TenantID,
	}}

	_, res.Err = c.Post(rootURL(c), reqBody, &res.Body, nil)
	return res
}

// Get retrieves a particular security group rule based on its unique ID.
func Get(c *gophercloud.ServiceClient, id string) GetResult {
	var res GetResult
	_, res.Err = c.Get(resourceURL(c, id), &res.Body, nil)
	return res
}

// Delete will permanently delete a particular security group rule based on its unique ID.
func Delete(c *gophercloud.ServiceClient, id string) DeleteResult {
	var res DeleteResult
	_, res.Err = c.Delete(resourceURL(c, id), nil)
	return res
}
