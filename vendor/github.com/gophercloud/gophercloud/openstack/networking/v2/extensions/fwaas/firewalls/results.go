package firewalls

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// Firewall is an OpenStack firewall.
type Firewall struct {
	ID           string `json:"id"`
	Name         string `json:"name"`
	Description  string `json:"description"`
	AdminStateUp bool   `json:"admin_state_up"`
	Status       string `json:"status"`
	PolicyID     string `json:"firewall_policy_id"`
	TenantID     string `json:"tenant_id"`
}

type commonResult struct {
	gophercloud.Result
}

// Extract is a function that accepts a result and extracts a firewall.
func (r commonResult) Extract() (*Firewall, error) {
	var s struct {
		Firewall *Firewall `json:"firewall"`
	}
	err := r.ExtractInto(&s)
	return s.Firewall, err
}

// FirewallPage is the page returned by a pager when traversing over a
// collection of firewalls.
type FirewallPage struct {
	pagination.LinkedPageBase
}

// NextPageURL is invoked when a paginated collection of firewalls has reached
// the end of a page and the pager seeks to traverse over a new one. In order
// to do this, it needs to construct the next page's URL.
func (r FirewallPage) NextPageURL() (string, error) {
	var s struct {
		Links []gophercloud.Link `json:"firewalls_links"`
	}
	err := r.ExtractInto(&s)
	if err != nil {
		return "", err
	}
	return gophercloud.ExtractNextURL(s.Links)
}

// IsEmpty checks whether a FirewallPage struct is empty.
func (r FirewallPage) IsEmpty() (bool, error) {
	is, err := ExtractFirewalls(r)
	return len(is) == 0, err
}

// ExtractFirewalls accepts a Page struct, specifically a RouterPage struct,
// and extracts the elements into a slice of Router structs. In other words,
// a generic collection is mapped into a relevant slice.
func ExtractFirewalls(r pagination.Page) ([]Firewall, error) {
	var s struct {
		Firewalls []Firewall `json:"firewalls" json:"firewalls"`
	}
	err := (r.(FirewallPage)).ExtractInto(&s)
	return s.Firewalls, err
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
