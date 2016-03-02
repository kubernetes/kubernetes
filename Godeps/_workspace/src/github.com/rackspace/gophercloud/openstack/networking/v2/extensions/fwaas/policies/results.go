package policies

import (
	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

type Policy struct {
	ID          string   `json:"id" mapstructure:"id"`
	Name        string   `json:"name" mapstructure:"name"`
	Description string   `json:"description" mapstructure:"description"`
	TenantID    string   `json:"tenant_id" mapstructure:"tenant_id"`
	Audited     bool     `json:"audited" mapstructure:"audited"`
	Shared      bool     `json:"shared" mapstructure:"shared"`
	Rules       []string `json:"firewall_rules,omitempty" mapstructure:"firewall_rules"`
}

type commonResult struct {
	gophercloud.Result
}

// Extract is a function that accepts a result and extracts a firewall policy.
func (r commonResult) Extract() (*Policy, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var res struct {
		Policy *Policy `json:"firewall_policy" mapstructure:"firewall_policy"`
	}

	err := mapstructure.Decode(r.Body, &res)

	return res.Policy, err
}

// PolicyPage is the page returned by a pager when traversing over a
// collection of firewall policies.
type PolicyPage struct {
	pagination.LinkedPageBase
}

// NextPageURL is invoked when a paginated collection of firewall policies has
// reached the end of a page and the pager seeks to traverse over a new one.
// In order to do this, it needs to construct the next page's URL.
func (p PolicyPage) NextPageURL() (string, error) {
	type resp struct {
		Links []gophercloud.Link `mapstructure:"firewall_policies_links"`
	}

	var r resp
	err := mapstructure.Decode(p.Body, &r)
	if err != nil {
		return "", err
	}

	return gophercloud.ExtractNextURL(r.Links)
}

// IsEmpty checks whether a PolicyPage struct is empty.
func (p PolicyPage) IsEmpty() (bool, error) {
	is, err := ExtractPolicies(p)
	if err != nil {
		return true, nil
	}
	return len(is) == 0, nil
}

// ExtractPolicies accepts a Page struct, specifically a RouterPage struct,
// and extracts the elements into a slice of Router structs. In other words,
// a generic collection is mapped into a relevant slice.
func ExtractPolicies(page pagination.Page) ([]Policy, error) {
	var resp struct {
		Policies []Policy `mapstructure:"firewall_policies" json:"firewall_policies"`
	}

	err := mapstructure.Decode(page.(PolicyPage).Body, &resp)

	return resp.Policies, err
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
