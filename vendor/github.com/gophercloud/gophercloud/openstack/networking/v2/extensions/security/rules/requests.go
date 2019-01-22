package rules

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// ListOpts allows the filtering and sorting of paginated collections through
// the API. Filtering is achieved by passing in struct field values that map to
// the security group rule attributes you want to see returned. SortKey allows
// you to sort by a particular network attribute. SortDir sets the direction,
// and is either `asc' or `desc'. Marker and Limit are used for pagination.
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
	ProjectID      string `q:"project_id"`
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

type RuleDirection string
type RuleProtocol string
type RuleEtherType string

// Constants useful for CreateOpts
const (
	DirIngress        RuleDirection = "ingress"
	DirEgress         RuleDirection = "egress"
	EtherType4        RuleEtherType = "IPv4"
	EtherType6        RuleEtherType = "IPv6"
	ProtocolAH        RuleProtocol  = "ah"
	ProtocolDCCP      RuleProtocol  = "dccp"
	ProtocolEGP       RuleProtocol  = "egp"
	ProtocolESP       RuleProtocol  = "esp"
	ProtocolGRE       RuleProtocol  = "gre"
	ProtocolICMP      RuleProtocol  = "icmp"
	ProtocolIGMP      RuleProtocol  = "igmp"
	ProtocolIPv6Encap RuleProtocol  = "ipv6-encap"
	ProtocolIPv6Frag  RuleProtocol  = "ipv6-frag"
	ProtocolIPv6ICMP  RuleProtocol  = "ipv6-icmp"
	ProtocolIPv6NoNxt RuleProtocol  = "ipv6-nonxt"
	ProtocolIPv6Opts  RuleProtocol  = "ipv6-opts"
	ProtocolIPv6Route RuleProtocol  = "ipv6-route"
	ProtocolOSPF      RuleProtocol  = "ospf"
	ProtocolPGM       RuleProtocol  = "pgm"
	ProtocolRSVP      RuleProtocol  = "rsvp"
	ProtocolSCTP      RuleProtocol  = "sctp"
	ProtocolTCP       RuleProtocol  = "tcp"
	ProtocolUDP       RuleProtocol  = "udp"
	ProtocolUDPLite   RuleProtocol  = "udplite"
	ProtocolVRRP      RuleProtocol  = "vrrp"
)

// CreateOptsBuilder allows extensions to add additional parameters to the
// Create request.
type CreateOptsBuilder interface {
	ToSecGroupRuleCreateMap() (map[string]interface{}, error)
}

// CreateOpts contains all the values needed to create a new security group
// rule.
type CreateOpts struct {
	// Must be either "ingress" or "egress": the direction in which the security
	// group rule is applied.
	Direction RuleDirection `json:"direction" required:"true"`

	// Must be "IPv4" or "IPv6", and addresses represented in CIDR must match the
	// ingress or egress rules.
	EtherType RuleEtherType `json:"ethertype" required:"true"`

	// The security group ID to associate with this security group rule.
	SecGroupID string `json:"security_group_id" required:"true"`

	// The maximum port number in the range that is matched by the security group
	// rule. The PortRangeMin attribute constrains the PortRangeMax attribute. If
	// the protocol is ICMP, this value must be an ICMP type.
	PortRangeMax int `json:"port_range_max,omitempty"`

	// The minimum port number in the range that is matched by the security group
	// rule. If the protocol is TCP or UDP, this value must be less than or equal
	// to the value of the PortRangeMax attribute. If the protocol is ICMP, this
	// value must be an ICMP type.
	PortRangeMin int `json:"port_range_min,omitempty"`

	// The protocol that is matched by the security group rule. Valid values are
	// "tcp", "udp", "icmp" or an empty string.
	Protocol RuleProtocol `json:"protocol,omitempty"`

	// The remote group ID to be associated with this security group rule. You can
	// specify either RemoteGroupID or RemoteIPPrefix.
	RemoteGroupID string `json:"remote_group_id,omitempty"`

	// The remote IP prefix to be associated with this security group rule. You can
	// specify either RemoteGroupID or RemoteIPPrefix. This attribute matches the
	// specified IP prefix as the source IP address of the IP packet.
	RemoteIPPrefix string `json:"remote_ip_prefix,omitempty"`

	// TenantID is the UUID of the project who owns the Rule.
	// Only administrative users can specify a project UUID other than their own.
	ProjectID string `json:"project_id,omitempty"`
}

// ToSecGroupRuleCreateMap builds a request body from CreateOpts.
func (opts CreateOpts) ToSecGroupRuleCreateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "security_group_rule")
}

// Create is an operation which adds a new security group rule and associates it
// with an existing security group (whose ID is specified in CreateOpts).
func Create(c *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	b, err := opts.ToSecGroupRuleCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = c.Post(rootURL(c), b, &r.Body, nil)
	return
}

// Get retrieves a particular security group rule based on its unique ID.
func Get(c *gophercloud.ServiceClient, id string) (r GetResult) {
	_, r.Err = c.Get(resourceURL(c, id), &r.Body, nil)
	return
}

// Delete will permanently delete a particular security group rule based on its
// unique ID.
func Delete(c *gophercloud.ServiceClient, id string) (r DeleteResult) {
	_, r.Err = c.Delete(resourceURL(c, id), nil)
	return
}
