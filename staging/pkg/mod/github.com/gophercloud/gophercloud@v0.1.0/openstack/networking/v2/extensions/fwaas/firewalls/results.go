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
	ProjectID    string `json:"project_id"`
}

type commonResult struct {
	gophercloud.Result
}

// Extract is a function that accepts a result and extracts a firewall.
func (r commonResult) Extract() (*Firewall, error) {
	var s Firewall
	err := r.ExtractInto(&s)
	return &s, err
}

func (r commonResult) ExtractInto(v interface{}) error {
	return r.Result.ExtractIntoStructPtr(v, "firewall")
}

func ExtractFirewallsInto(r pagination.Page, v interface{}) error {
	return r.(FirewallPage).Result.ExtractIntoSlicePtr(v, "firewalls")
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

// ExtractFirewalls accepts a Page struct, specifically a FirewallPage struct,
// and extracts the elements into a slice of Firewall structs. In other words,
// a generic collection is mapped into a relevant slice.
func ExtractFirewalls(r pagination.Page) ([]Firewall, error) {
	var s []Firewall
	err := ExtractFirewallsInto(r, &s)
	return s, err
}

// GetResult represents the result of a Get operation. Call its Extract
// method to interpret it as a Firewall.
type GetResult struct {
	commonResult
}

// UpdateResult represents the result of an Update operation. Call its Extract
// method to interpret it as a Firewall.
type UpdateResult struct {
	commonResult
}

// DeleteResult represents the result of a delete operation. Call its
// ExtractErr method to determine if the operation succeeded or failed.
type DeleteResult struct {
	gophercloud.ErrResult
}

// CreateResult represents the result of a Create operation. Call its Extract
// method to interpret it as a Firewall.
type CreateResult struct {
	commonResult
}
