package rules

import (
	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// SecGroupRule represents a rule to dictate the behaviour of incoming or
// outgoing traffic for a particular security group.
type SecGroupRule struct {
	// The UUID for this security group rule.
	ID string

	// The direction in which the security group rule is applied. The only values
	// allowed are "ingress" or "egress". For a compute instance, an ingress
	// security group rule is applied to incoming (ingress) traffic for that
	// instance. An egress rule is applied to traffic leaving the instance.
	Direction string

	// Must be IPv4 or IPv6, and addresses represented in CIDR must match the
	// ingress or egress rules.
	EtherType string `json:"ethertype" mapstructure:"ethertype"`

	// The security group ID to associate with this security group rule.
	SecGroupID string `json:"security_group_id" mapstructure:"security_group_id"`

	// The minimum port number in the range that is matched by the security group
	// rule. If the protocol is TCP or UDP, this value must be less than or equal
	// to the value of the PortRangeMax attribute. If the protocol is ICMP, this
	// value must be an ICMP type.
	PortRangeMin int `json:"port_range_min" mapstructure:"port_range_min"`

	// The maximum port number in the range that is matched by the security group
	// rule. The PortRangeMin attribute constrains the PortRangeMax attribute. If
	// the protocol is ICMP, this value must be an ICMP type.
	PortRangeMax int `json:"port_range_max" mapstructure:"port_range_max"`

	// The protocol that is matched by the security group rule. Valid values are
	// "tcp", "udp", "icmp" or an empty string.
	Protocol string

	// The remote group ID to be associated with this security group rule. You
	// can specify either RemoteGroupID or RemoteIPPrefix.
	RemoteGroupID string `json:"remote_group_id" mapstructure:"remote_group_id"`

	// The remote IP prefix to be associated with this security group rule. You
	// can specify either RemoteGroupID or RemoteIPPrefix . This attribute
	// matches the specified IP prefix as the source IP address of the IP packet.
	RemoteIPPrefix string `json:"remote_ip_prefix" mapstructure:"remote_ip_prefix"`

	// The owner of this security group rule.
	TenantID string `json:"tenant_id" mapstructure:"tenant_id"`
}

// SecGroupRulePage is the page returned by a pager when traversing over a
// collection of security group rules.
type SecGroupRulePage struct {
	pagination.LinkedPageBase
}

// NextPageURL is invoked when a paginated collection of security group rules has
// reached the end of a page and the pager seeks to traverse over a new one. In
// order to do this, it needs to construct the next page's URL.
func (p SecGroupRulePage) NextPageURL() (string, error) {
	type resp struct {
		Links []gophercloud.Link `mapstructure:"security_group_rules_links"`
	}

	var r resp
	err := mapstructure.Decode(p.Body, &r)
	if err != nil {
		return "", err
	}

	return gophercloud.ExtractNextURL(r.Links)
}

// IsEmpty checks whether a SecGroupRulePage struct is empty.
func (p SecGroupRulePage) IsEmpty() (bool, error) {
	is, err := ExtractRules(p)
	if err != nil {
		return true, nil
	}
	return len(is) == 0, nil
}

// ExtractRules accepts a Page struct, specifically a SecGroupRulePage struct,
// and extracts the elements into a slice of SecGroupRule structs. In other words,
// a generic collection is mapped into a relevant slice.
func ExtractRules(page pagination.Page) ([]SecGroupRule, error) {
	var resp struct {
		SecGroupRules []SecGroupRule `mapstructure:"security_group_rules" json:"security_group_rules"`
	}

	err := mapstructure.Decode(page.(SecGroupRulePage).Body, &resp)

	return resp.SecGroupRules, err
}

type commonResult struct {
	gophercloud.Result
}

// Extract is a function that accepts a result and extracts a security rule.
func (r commonResult) Extract() (*SecGroupRule, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var res struct {
		SecGroupRule *SecGroupRule `mapstructure:"security_group_rule" json:"security_group_rule"`
	}

	err := mapstructure.Decode(r.Body, &res)

	return res.SecGroupRule, err
}

// CreateResult represents the result of a create operation.
type CreateResult struct {
	commonResult
}

// GetResult represents the result of a get operation.
type GetResult struct {
	commonResult
}

// DeleteResult represents the result of a delete operation.
type DeleteResult struct {
	gophercloud.ErrResult
}
