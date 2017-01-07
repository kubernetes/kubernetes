package rules

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
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

type RuleDirection string
type RuleProtocol string
type RuleEtherType string

// Constants useful for CreateOpts
const (
	DirIngress   RuleDirection = "ingress"
	DirEgress    RuleDirection = "egress"
	ProtocolTCP  RuleProtocol  = "tcp"
	ProtocolUDP  RuleProtocol  = "udp"
	ProtocolICMP RuleProtocol  = "icmp"
	EtherType4   RuleEtherType = "IPv4"
	EtherType6   RuleEtherType = "IPv6"
)

// CreateOptsBuilder is what types must satisfy to be used as Create
// options.
type CreateOptsBuilder interface {
	ToSecGroupRuleCreateMap() (map[string]interface{}, error)
}

// CreateOpts contains all the values needed to create a new security group rule.
type CreateOpts struct {
	// Required. Must be either "ingress" or "egress": the direction in which the
	// security group rule is applied.
	Direction RuleDirection `json:"direction" required:"true"`
	// Required. Must be "IPv4" or "IPv6", and addresses represented in CIDR must
	// match the ingress or egress rules.
	EtherType RuleEtherType `json:"ethertype" required:"true"`
	// Required. The security group ID to associate with this security group rule.
	SecGroupID string `json:"security_group_id" required:"true"`
	// Optional. The maximum port number in the range that is matched by the
	// security group rule. The PortRangeMin attribute constrains the PortRangeMax
	// attribute. If the protocol is ICMP, this value must be an ICMP type.
	PortRangeMax int `json:"port_range_max,omitempty"`
	// Optional. The minimum port number in the range that is matched by the
	// security group rule. If the protocol is TCP or UDP, this value must be
	// less than or equal to the value of the PortRangeMax attribute. If the
	// protocol is ICMP, this value must be an ICMP type.
	PortRangeMin int `json:"port_range_min,omitempty"`
	// Optional. The protocol that is matched by the security group rule. Valid
	// values are "tcp", "udp", "icmp" or an empty string.
	Protocol RuleProtocol `json:"protocol,omitempty"`
	// Optional. The remote group ID to be associated with this security group
	// rule. You can specify either RemoteGroupID or RemoteIPPrefix.
	RemoteGroupID string `json:"remote_group_id,omitempty"`
	// Optional. The remote IP prefix to be associated with this security group
	// rule. You can specify either RemoteGroupID or RemoteIPPrefix. This
	// attribute matches the specified IP prefix as the source IP address of the
	// IP packet.
	RemoteIPPrefix string `json:"remote_ip_prefix,omitempty"`
	// Required for admins. Indicates the owner of the VIP.
	TenantID string `json:"tenant_id,omitempty"`
}

// ToSecGroupRuleCreateMap allows CreateOpts to satisfy the CreateOptsBuilder
// interface
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

// Delete will permanently delete a particular security group rule based on its unique ID.
func Delete(c *gophercloud.ServiceClient, id string) (r DeleteResult) {
	_, r.Err = c.Delete(resourceURL(c, id), nil)
	return
}
