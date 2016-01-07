package defsecrules

import (
	"github.com/mitchellh/mapstructure"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/openstack/compute/v2/extensions/secgroups"
	"github.com/rackspace/gophercloud/pagination"
)

// DefaultRule represents a default rule - which is identical to a
// normal security rule.
type DefaultRule secgroups.Rule

// DefaultRulePage is a single page of a DefaultRule collection.
type DefaultRulePage struct {
	pagination.SinglePageBase
}

// IsEmpty determines whether or not a page of default rules contains any results.
func (page DefaultRulePage) IsEmpty() (bool, error) {
	users, err := ExtractDefaultRules(page)
	if err != nil {
		return false, err
	}
	return len(users) == 0, nil
}

// ExtractDefaultRules returns a slice of DefaultRules contained in a single
// page of results.
func ExtractDefaultRules(page pagination.Page) ([]DefaultRule, error) {
	casted := page.(DefaultRulePage).Body
	var response struct {
		Rules []DefaultRule `mapstructure:"security_group_default_rules"`
	}

	err := mapstructure.WeakDecode(casted, &response)

	return response.Rules, err
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

// Extract will extract a DefaultRule struct from most responses.
func (r commonResult) Extract() (*DefaultRule, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var response struct {
		Rule DefaultRule `mapstructure:"security_group_default_rule"`
	}

	err := mapstructure.WeakDecode(r.Body, &response)

	return &response.Rule, err
}
