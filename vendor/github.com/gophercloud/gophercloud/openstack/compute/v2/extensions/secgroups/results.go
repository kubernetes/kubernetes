package secgroups

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// SecurityGroup represents a security group.
type SecurityGroup struct {
	// The unique ID of the group. If Neutron is installed, this ID will be
	// represented as a string UUID; if Neutron is not installed, it will be a
	// numeric ID. For the sake of consistency, we always cast it to a string.
	ID string

	// The human-readable name of the group, which needs to be unique.
	Name string

	// The human-readable description of the group.
	Description string

	// The rules which determine how this security group operates.
	Rules []Rule

	// The ID of the tenant to which this security group belongs.
	TenantID string `json:"tenant_id"`
}

// Rule represents a security group rule, a policy which determines how a
// security group operates and what inbound traffic it allows in.
type Rule struct {
	// The unique ID. If Neutron is installed, this ID will be
	// represented as a string UUID; if Neutron is not installed, it will be a
	// numeric ID. For the sake of consistency, we always cast it to a string.
	ID string

	// The lower bound of the port range which this security group should open up
	FromPort int `json:"from_port"`

	// The upper bound of the port range which this security group should open up
	ToPort int `json:"to_port"`

	// The IP protocol (e.g. TCP) which the security group accepts
	IPProtocol string `json:"ip_protocol"`

	// The CIDR IP range whose traffic can be received
	IPRange IPRange `json:"ip_range"`

	// The security group ID to which this rule belongs
	ParentGroupID string `json:"parent_group_id"`

	// Not documented.
	Group Group
}

// IPRange represents the IP range whose traffic will be accepted by the
// security group.
type IPRange struct {
	CIDR string
}

// Group represents a group.
type Group struct {
	TenantID string `json:"tenant_id"`
	Name     string
}

// SecurityGroupPage is a single page of a SecurityGroup collection.
type SecurityGroupPage struct {
	pagination.SinglePageBase
}

// IsEmpty determines whether or not a page of Security Groups contains any results.
func (page SecurityGroupPage) IsEmpty() (bool, error) {
	users, err := ExtractSecurityGroups(page)
	return len(users) == 0, err
}

// ExtractSecurityGroups returns a slice of SecurityGroups contained in a single page of results.
func ExtractSecurityGroups(r pagination.Page) ([]SecurityGroup, error) {
	var s struct {
		SecurityGroups []SecurityGroup `json:"security_groups"`
	}
	err := (r.(SecurityGroupPage)).ExtractInto(&s)
	return s.SecurityGroups, err
}

type commonResult struct {
	gophercloud.Result
}

// CreateResult represents the result of a create operation.
type CreateResult struct {
	commonResult
}

// GetResult represents the result of a get operation.
type GetResult struct {
	commonResult
}

// UpdateResult represents the result of an update operation.
type UpdateResult struct {
	commonResult
}

// Extract will extract a SecurityGroup struct from most responses.
func (r commonResult) Extract() (*SecurityGroup, error) {
	var s struct {
		SecurityGroup *SecurityGroup `json:"security_group"`
	}
	err := r.ExtractInto(&s)
	return s.SecurityGroup, err
}

// CreateRuleResult represents the result when adding rules to a security group.
type CreateRuleResult struct {
	gophercloud.Result
}

// Extract will extract a Rule struct from a CreateRuleResult.
func (r CreateRuleResult) Extract() (*Rule, error) {
	var s struct {
		Rule *Rule `json:"security_group_rule"`
	}
	err := r.ExtractInto(&s)
	return s.Rule, err
}
