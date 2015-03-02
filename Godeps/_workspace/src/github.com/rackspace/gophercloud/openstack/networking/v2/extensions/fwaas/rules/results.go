package rules

import (
	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// Rule represents a firewall rule
type Rule struct {
	ID                   string `json:"id" mapstructure:"id"`
	Name                 string `json:"name,omitempty" mapstructure:"name"`
	Description          string `json:"description,omitempty" mapstructure:"description"`
	Protocol             string `json:"protocol" mapstructure:"protocol"`
	Action               string `json:"action" mapstructure:"action"`
	IPVersion            int    `json:"ip_version,omitempty" mapstructure:"ip_version"`
	SourceIPAddress      string `json:"source_ip_address,omitempty" mapstructure:"source_ip_address"`
	DestinationIPAddress string `json:"destination_ip_address,omitempty" mapstructure:"destination_ip_address"`
	SourcePort           string `json:"source_port,omitempty" mapstructure:"source_port"`
	DestinationPort      string `json:"destination_port,omitempty" mapstructure:"destination_port"`
	Shared               bool   `json:"shared,omitempty" mapstructure:"shared"`
	Enabled              bool   `json:"enabled,omitempty" mapstructure:"enabled"`
	PolicyID             string `json:"firewall_policy_id" mapstructure:"firewall_policy_id"`
	Position             int    `json:"position" mapstructure:"position"`
	TenantID             string `json:"tenant_id" mapstructure:"tenant_id"`
}

// RulePage is the page returned by a pager when traversing over a
// collection of firewall rules.
type RulePage struct {
	pagination.LinkedPageBase
}

// NextPageURL is invoked when a paginated collection of firewall rules has
// reached the end of a page and the pager seeks to traverse over a new one.
// In order to do this, it needs to construct the next page's URL.
func (p RulePage) NextPageURL() (string, error) {
	type resp struct {
		Links []gophercloud.Link `mapstructure:"firewall_rules_links"`
	}

	var r resp
	err := mapstructure.Decode(p.Body, &r)
	if err != nil {
		return "", err
	}

	return gophercloud.ExtractNextURL(r.Links)
}

// IsEmpty checks whether a RulePage struct is empty.
func (p RulePage) IsEmpty() (bool, error) {
	is, err := ExtractRules(p)
	if err != nil {
		return true, nil
	}
	return len(is) == 0, nil
}

// ExtractRules accepts a Page struct, specifically a RouterPage struct,
// and extracts the elements into a slice of Router structs. In other words,
// a generic collection is mapped into a relevant slice.
func ExtractRules(page pagination.Page) ([]Rule, error) {
	var resp struct {
		Rules []Rule `mapstructure:"firewall_rules" json:"firewall_rules"`
	}

	err := mapstructure.Decode(page.(RulePage).Body, &resp)

	return resp.Rules, err
}

type commonResult struct {
	gophercloud.Result
}

// Extract is a function that accepts a result and extracts a firewall rule.
func (r commonResult) Extract() (*Rule, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var res struct {
		Rule *Rule `json:"firewall_rule" mapstructure:"firewall_rule"`
	}

	err := mapstructure.Decode(r.Body, &res)

	return res.Rule, err
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
