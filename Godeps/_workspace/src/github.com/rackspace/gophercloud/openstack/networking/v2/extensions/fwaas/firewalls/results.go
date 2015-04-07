package firewalls

import (
	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

type Firewall struct {
	ID           string `json:"id" mapstructure:"id"`
	Name         string `json:"name" mapstructure:"name"`
	Description  string `json:"description" mapstructure:"description"`
	AdminStateUp bool   `json:"admin_state_up" mapstructure:"admin_state_up"`
	Status       string `json:"status" mapstructure:"status"`
	PolicyID     string `json:"firewall_policy_id" mapstructure:"firewall_policy_id"`
	TenantID     string `json:"tenant_id" mapstructure:"tenant_id"`
}

type commonResult struct {
	gophercloud.Result
}

// Extract is a function that accepts a result and extracts a firewall.
func (r commonResult) Extract() (*Firewall, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var res struct {
		Firewall *Firewall `json:"firewall"`
	}

	err := mapstructure.Decode(r.Body, &res)

	return res.Firewall, err
}

// FirewallPage is the page returned by a pager when traversing over a
// collection of firewalls.
type FirewallPage struct {
	pagination.LinkedPageBase
}

// NextPageURL is invoked when a paginated collection of firewalls has reached
// the end of a page and the pager seeks to traverse over a new one. In order
// to do this, it needs to construct the next page's URL.
func (p FirewallPage) NextPageURL() (string, error) {
	type resp struct {
		Links []gophercloud.Link `mapstructure:"firewalls_links"`
	}

	var r resp
	err := mapstructure.Decode(p.Body, &r)
	if err != nil {
		return "", err
	}

	return gophercloud.ExtractNextURL(r.Links)
}

// IsEmpty checks whether a FirewallPage struct is empty.
func (p FirewallPage) IsEmpty() (bool, error) {
	is, err := ExtractFirewalls(p)
	if err != nil {
		return true, nil
	}
	return len(is) == 0, nil
}

// ExtractFirewalls accepts a Page struct, specifically a RouterPage struct,
// and extracts the elements into a slice of Router structs. In other words,
// a generic collection is mapped into a relevant slice.
func ExtractFirewalls(page pagination.Page) ([]Firewall, error) {
	var resp struct {
		Firewalls []Firewall `mapstructure:"firewalls" json:"firewalls"`
	}

	err := mapstructure.Decode(page.(FirewallPage).Body, &resp)

	return resp.Firewalls, err
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
