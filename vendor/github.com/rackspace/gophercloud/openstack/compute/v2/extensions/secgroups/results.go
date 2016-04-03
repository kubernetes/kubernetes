package secgroups

import (
	"github.com/mitchellh/mapstructure"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
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
	TenantID string `mapstructure:"tenant_id"`
}

// Rule represents a security group rule, a policy which determines how a
// security group operates and what inbound traffic it allows in.
type Rule struct {
	// The unique ID. If Neutron is installed, this ID will be
	// represented as a string UUID; if Neutron is not installed, it will be a
	// numeric ID. For the sake of consistency, we always cast it to a string.
	ID string

	// The lower bound of the port range which this security group should open up
	FromPort int `mapstructure:"from_port"`

	// The upper bound of the port range which this security group should open up
	ToPort int `mapstructure:"to_port"`

	// The IP protocol (e.g. TCP) which the security group accepts
	IPProtocol string `mapstructure:"ip_protocol"`

	// The CIDR IP range whose traffic can be received
	IPRange IPRange `mapstructure:"ip_range"`

	// The security group ID to which this rule belongs
	ParentGroupID string `mapstructure:"parent_group_id"`

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
	TenantID string `mapstructure:"tenant_id"`
	Name     string
}

// SecurityGroupPage is a single page of a SecurityGroup collection.
type SecurityGroupPage struct {
	pagination.SinglePageBase
}

// IsEmpty determines whether or not a page of Security Groups contains any results.
func (page SecurityGroupPage) IsEmpty() (bool, error) {
	users, err := ExtractSecurityGroups(page)
	if err != nil {
		return false, err
	}
	return len(users) == 0, nil
}

// ExtractSecurityGroups returns a slice of SecurityGroups contained in a single page of results.
func ExtractSecurityGroups(page pagination.Page) ([]SecurityGroup, error) {
	casted := page.(SecurityGroupPage).Body
	var response struct {
		SecurityGroups []SecurityGroup `mapstructure:"security_groups"`
	}

	err := mapstructure.WeakDecode(casted, &response)

	return response.SecurityGroups, err
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
	if r.Err != nil {
		return nil, r.Err
	}

	var response struct {
		SecurityGroup SecurityGroup `mapstructure:"security_group"`
	}

	err := mapstructure.WeakDecode(r.Body, &response)

	return &response.SecurityGroup, err
}

// CreateRuleResult represents the result when adding rules to a security group.
type CreateRuleResult struct {
	gophercloud.Result
}

// Extract will extract a Rule struct from a CreateRuleResult.
func (r CreateRuleResult) Extract() (*Rule, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var response struct {
		Rule Rule `mapstructure:"security_group_rule"`
	}

	err := mapstructure.WeakDecode(r.Body, &response)

	return &response.Rule, err
}
