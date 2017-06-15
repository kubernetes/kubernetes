package groups

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/security/rules"
	"github.com/gophercloud/gophercloud/pagination"
)

// SecGroup represents a container for security group rules.
type SecGroup struct {
	// The UUID for the security group.
	ID string

	// Human-readable name for the security group. Might not be unique. Cannot be
	// named "default" as that is automatically created for a tenant.
	Name string

	// The security group description.
	Description string

	// A slice of security group rules that dictate the permitted behaviour for
	// traffic entering and leaving the group.
	Rules []rules.SecGroupRule `json:"security_group_rules"`

	// Owner of the security group. Only admin users can specify a TenantID
	// other than their own.
	TenantID string `json:"tenant_id"`
}

// SecGroupPage is the page returned by a pager when traversing over a
// collection of security groups.
type SecGroupPage struct {
	pagination.LinkedPageBase
}

// NextPageURL is invoked when a paginated collection of security groups has
// reached the end of a page and the pager seeks to traverse over a new one. In
// order to do this, it needs to construct the next page's URL.
func (r SecGroupPage) NextPageURL() (string, error) {
	var s struct {
		Links []gophercloud.Link `json:"security_groups_links"`
	}
	err := r.ExtractInto(&s)
	if err != nil {
		return "", err
	}

	return gophercloud.ExtractNextURL(s.Links)
}

// IsEmpty checks whether a SecGroupPage struct is empty.
func (r SecGroupPage) IsEmpty() (bool, error) {
	is, err := ExtractGroups(r)
	return len(is) == 0, err
}

// ExtractGroups accepts a Page struct, specifically a SecGroupPage struct,
// and extracts the elements into a slice of SecGroup structs. In other words,
// a generic collection is mapped into a relevant slice.
func ExtractGroups(r pagination.Page) ([]SecGroup, error) {
	var s struct {
		SecGroups []SecGroup `json:"security_groups"`
	}
	err := (r.(SecGroupPage)).ExtractInto(&s)
	return s.SecGroups, err
}

type commonResult struct {
	gophercloud.Result
}

// Extract is a function that accepts a result and extracts a security group.
func (r commonResult) Extract() (*SecGroup, error) {
	var s struct {
		SecGroup *SecGroup `json:"security_group"`
	}
	err := r.ExtractInto(&s)
	return s.SecGroup, err
}

// CreateResult represents the result of a create operation.
type CreateResult struct {
	commonResult
}

// UpdateResult represents the result of an update operation.
type UpdateResult struct {
	commonResult
}

// GetResult represents the result of a get operation.
type GetResult struct {
	commonResult
}

// DeleteResult represents the result of a delete operation.
type DeleteResult struct {
	gophercloud.ErrResult
}
