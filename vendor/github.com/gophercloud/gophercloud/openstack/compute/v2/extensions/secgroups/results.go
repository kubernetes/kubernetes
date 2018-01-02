package secgroups

import (
	"encoding/json"
	"strconv"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// SecurityGroup represents a security group.
type SecurityGroup struct {
	// The unique ID of the group. If Neutron is installed, this ID will be
	// represented as a string UUID; if Neutron is not installed, it will be a
	// numeric ID. For the sake of consistency, we always cast it to a string.
	ID string `json:"-"`

	// The human-readable name of the group, which needs to be unique.
	Name string `json:"name"`

	// The human-readable description of the group.
	Description string `json:"description"`

	// The rules which determine how this security group operates.
	Rules []Rule `json:"rules"`

	// The ID of the tenant to which this security group belongs.
	TenantID string `json:"tenant_id"`
}

func (r *SecurityGroup) UnmarshalJSON(b []byte) error {
	type tmp SecurityGroup
	var s struct {
		tmp
		ID interface{} `json:"id"`
	}
	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}

	*r = SecurityGroup(s.tmp)

	switch t := s.ID.(type) {
	case float64:
		r.ID = strconv.FormatFloat(t, 'f', -1, 64)
	case string:
		r.ID = t
	}

	return err
}

// Rule represents a security group rule, a policy which determines how a
// security group operates and what inbound traffic it allows in.
type Rule struct {
	// The unique ID. If Neutron is installed, this ID will be
	// represented as a string UUID; if Neutron is not installed, it will be a
	// numeric ID. For the sake of consistency, we always cast it to a string.
	ID string `json:"-"`

	// The lower bound of the port range which this security group should open up.
	FromPort int `json:"from_port"`

	// The upper bound of the port range which this security group should open up.
	ToPort int `json:"to_port"`

	// The IP protocol (e.g. TCP) which the security group accepts.
	IPProtocol string `json:"ip_protocol"`

	// The CIDR IP range whose traffic can be received.
	IPRange IPRange `json:"ip_range"`

	// The security group ID to which this rule belongs.
	ParentGroupID string `json:"parent_group_id"`

	// Not documented.
	Group Group
}

func (r *Rule) UnmarshalJSON(b []byte) error {
	type tmp Rule
	var s struct {
		tmp
		ID            interface{} `json:"id"`
		ParentGroupID interface{} `json:"parent_group_id"`
	}
	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}

	*r = Rule(s.tmp)

	switch t := s.ID.(type) {
	case float64:
		r.ID = strconv.FormatFloat(t, 'f', -1, 64)
	case string:
		r.ID = t
	}

	switch t := s.ParentGroupID.(type) {
	case float64:
		r.ParentGroupID = strconv.FormatFloat(t, 'f', -1, 64)
	case string:
		r.ParentGroupID = t
	}

	return err
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

// IsEmpty determines whether or not a page of Security Groups contains any
// results.
func (page SecurityGroupPage) IsEmpty() (bool, error) {
	users, err := ExtractSecurityGroups(page)
	return len(users) == 0, err
}

// ExtractSecurityGroups returns a slice of SecurityGroups contained in a
// single page of results.
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

// CreateResult represents the result of a create operation. Call its Extract
// method to interpret the result as a SecurityGroup.
type CreateResult struct {
	commonResult
}

// GetResult represents the result of a get operation. Call its Extract
// method to interpret the result as a SecurityGroup.
type GetResult struct {
	commonResult
}

// UpdateResult represents the result of an update operation. Call its Extract
// method to interpret the result as a SecurityGroup.
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
// Call its Extract method to interpret the result as a Rule.
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

// DeleteResult is the response from delete operation. Call its ExtractErr
// method to determine if the request succeeded or failed.
type DeleteResult struct {
	gophercloud.ErrResult
}

// DeleteRuleResult is the response from a DeleteRule operation. Call its
// ExtractErr method to determine if the request succeeded or failed.
type DeleteRuleResult struct {
	gophercloud.ErrResult
}

// AddServerResult is the response from an AddServer operation. Call its
// ExtractErr method to determine if the request succeeded or failed.
type AddServerResult struct {
	gophercloud.ErrResult
}

// RemoveServerResult is the response from a RemoveServer operation. Call its
// ExtractErr method to determine if the request succeeded or failed.
type RemoveServerResult struct {
	gophercloud.ErrResult
}
