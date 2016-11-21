package defsecrules

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/secgroups"
	"github.com/gophercloud/gophercloud/pagination"
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
	return len(users) == 0, err
}

// ExtractDefaultRules returns a slice of DefaultRules contained in a single
// page of results.
func ExtractDefaultRules(r pagination.Page) ([]DefaultRule, error) {
	var s struct {
		DefaultRules []DefaultRule `json:"security_group_default_rules"`
	}
	err := (r.(DefaultRulePage)).ExtractInto(&s)
	return s.DefaultRules, err
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
	var s struct {
		DefaultRule DefaultRule `json:"security_group_default_rule"`
	}
	err := r.ExtractInto(&s)
	return &s.DefaultRule, err
}
