package rules

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// Rule represents a firewall rule
type Rule struct {
	ID                   string `json:"id"`
	Name                 string `json:"name,omitempty"`
	Description          string `json:"description,omitempty"`
	Protocol             string `json:"protocol"`
	Action               string `json:"action"`
	IPVersion            int    `json:"ip_version,omitempty"`
	SourceIPAddress      string `json:"source_ip_address,omitempty"`
	DestinationIPAddress string `json:"destination_ip_address,omitempty"`
	SourcePort           string `json:"source_port,omitempty"`
	DestinationPort      string `json:"destination_port,omitempty"`
	Shared               bool   `json:"shared,omitempty"`
	Enabled              bool   `json:"enabled,omitempty"`
	PolicyID             string `json:"firewall_policy_id"`
	Position             int    `json:"position"`
	TenantID             string `json:"tenant_id"`
}

// RulePage is the page returned by a pager when traversing over a
// collection of firewall rules.
type RulePage struct {
	pagination.LinkedPageBase
}

// NextPageURL is invoked when a paginated collection of firewall rules has
// reached the end of a page and the pager seeks to traverse over a new one.
// In order to do this, it needs to construct the next page's URL.
func (r RulePage) NextPageURL() (string, error) {
	var s struct {
		Links []gophercloud.Link `json:"firewall_rules_links"`
	}
	err := r.ExtractInto(&s)
	if err != nil {
		return "", err
	}
	return gophercloud.ExtractNextURL(s.Links)
}

// IsEmpty checks whether a RulePage struct is empty.
func (r RulePage) IsEmpty() (bool, error) {
	is, err := ExtractRules(r)
	return len(is) == 0, err
}

// ExtractRules accepts a Page struct, specifically a RouterPage struct,
// and extracts the elements into a slice of Router structs. In other words,
// a generic collection is mapped into a relevant slice.
func ExtractRules(r pagination.Page) ([]Rule, error) {
	var s struct {
		Rules []Rule `json:"firewall_rules"`
	}
	err := (r.(RulePage)).ExtractInto(&s)
	return s.Rules, err
}

type commonResult struct {
	gophercloud.Result
}

// Extract is a function that accepts a result and extracts a firewall rule.
func (r commonResult) Extract() (*Rule, error) {
	var s struct {
		Rule *Rule `json:"firewall_rule"`
	}
	err := r.ExtractInto(&s)
	return s.Rule, err
}

// GetResult represents the result of a get operation.
type GetResult struct {
	commonResult
}

// UpdateResult represents the result of an update operation.
type UpdateResult struct {
	commonResult
}

// DeleteResult represents the result of a delete operation.
type DeleteResult struct {
	gophercloud.ErrResult
}

// CreateResult represents the result of a create operation.
type CreateResult struct {
	commonResult
}
