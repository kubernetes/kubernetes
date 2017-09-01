package rules

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
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
	EtherType string `json:"ethertype"`

	// The security group ID to associate with this security group rule.
	SecGroupID string `json:"security_group_id"`

	// The minimum port number in the range that is matched by the security group
	// rule. If the protocol is TCP or UDP, this value must be less than or equal
	// to the value of the PortRangeMax attribute. If the protocol is ICMP, this
	// value must be an ICMP type.
	PortRangeMin int `json:"port_range_min"`

	// The maximum port number in the range that is matched by the security group
	// rule. The PortRangeMin attribute constrains the PortRangeMax attribute. If
	// the protocol is ICMP, this value must be an ICMP type.
	PortRangeMax int `json:"port_range_max"`

	// The protocol that is matched by the security group rule. Valid values are
	// "tcp", "udp", "icmp" or an empty string.
	Protocol string

	// The remote group ID to be associated with this security group rule. You
	// can specify either RemoteGroupID or RemoteIPPrefix.
	RemoteGroupID string `json:"remote_group_id"`

	// The remote IP prefix to be associated with this security group rule. You
	// can specify either RemoteGroupID or RemoteIPPrefix . This attribute
	// matches the specified IP prefix as the source IP address of the IP packet.
	RemoteIPPrefix string `json:"remote_ip_prefix"`

	// The owner of this security group rule.
	TenantID string `json:"tenant_id"`
}

// SecGroupRulePage is the page returned by a pager when traversing over a
// collection of security group rules.
type SecGroupRulePage struct {
	pagination.LinkedPageBase
}

// NextPageURL is invoked when a paginated collection of security group rules has
// reached the end of a page and the pager seeks to traverse over a new one. In
// order to do this, it needs to construct the next page's URL.
func (r SecGroupRulePage) NextPageURL() (string, error) {
	var s struct {
		Links []gophercloud.Link `json:"security_group_rules_links"`
	}
	err := r.ExtractInto(&s)
	if err != nil {
		return "", err
	}
	return gophercloud.ExtractNextURL(s.Links)
}

// IsEmpty checks whether a SecGroupRulePage struct is empty.
func (r SecGroupRulePage) IsEmpty() (bool, error) {
	is, err := ExtractRules(r)
	return len(is) == 0, err
}

// ExtractRules accepts a Page struct, specifically a SecGroupRulePage struct,
// and extracts the elements into a slice of SecGroupRule structs. In other words,
// a generic collection is mapped into a relevant slice.
func ExtractRules(r pagination.Page) ([]SecGroupRule, error) {
	var s struct {
		SecGroupRules []SecGroupRule `json:"security_group_rules"`
	}
	err := (r.(SecGroupRulePage)).ExtractInto(&s)
	return s.SecGroupRules, err
}

type commonResult struct {
	gophercloud.Result
}

// Extract is a function that accepts a result and extracts a security rule.
func (r commonResult) Extract() (*SecGroupRule, error) {
	var s struct {
		SecGroupRule *SecGroupRule `json:"security_group_rule"`
	}
	err := r.ExtractInto(&s)
	return s.SecGroupRule, err
}

// CreateResult represents the result of a create operation. Call its Extract
// method to interpret it as a SecGroupRule.
type CreateResult struct {
	commonResult
}

// GetResult represents the result of a get operation. Call its Extract
// method to interpret it as a SecGroupRule.
type GetResult struct {
	commonResult
}

// DeleteResult represents the result of a delete operation. Call its
// ExtractErr method to determine if the request succeeded or failed.
type DeleteResult struct {
	gophercloud.ErrResult
}
