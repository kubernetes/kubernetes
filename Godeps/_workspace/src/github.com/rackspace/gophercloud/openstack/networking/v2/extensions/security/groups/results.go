package groups

import (
	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/openstack/networking/v2/extensions/security/rules"
	"github.com/rackspace/gophercloud/pagination"
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
	Rules []rules.SecGroupRule `json:"security_group_rules" mapstructure:"security_group_rules"`

	// Owner of the security group. Only admin users can specify a TenantID
	// other than their own.
	TenantID string `json:"tenant_id" mapstructure:"tenant_id"`
}

// SecGroupPage is the page returned by a pager when traversing over a
// collection of security groups.
type SecGroupPage struct {
	pagination.LinkedPageBase
}

// NextPageURL is invoked when a paginated collection of security groups has
// reached the end of a page and the pager seeks to traverse over a new one. In
// order to do this, it needs to construct the next page's URL.
func (p SecGroupPage) NextPageURL() (string, error) {
	type resp struct {
		Links []gophercloud.Link `mapstructure:"security_groups_links"`
	}

	var r resp
	err := mapstructure.Decode(p.Body, &r)
	if err != nil {
		return "", err
	}

	return gophercloud.ExtractNextURL(r.Links)
}

// IsEmpty checks whether a SecGroupPage struct is empty.
func (p SecGroupPage) IsEmpty() (bool, error) {
	is, err := ExtractGroups(p)
	if err != nil {
		return true, nil
	}
	return len(is) == 0, nil
}

// ExtractGroups accepts a Page struct, specifically a SecGroupPage struct,
// and extracts the elements into a slice of SecGroup structs. In other words,
// a generic collection is mapped into a relevant slice.
func ExtractGroups(page pagination.Page) ([]SecGroup, error) {
	var resp struct {
		SecGroups []SecGroup `mapstructure:"security_groups" json:"security_groups"`
	}

	err := mapstructure.Decode(page.(SecGroupPage).Body, &resp)

	return resp.SecGroups, err
}

type commonResult struct {
	gophercloud.Result
}

// Extract is a function that accepts a result and extracts a security group.
func (r commonResult) Extract() (*SecGroup, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var res struct {
		SecGroup *SecGroup `mapstructure:"security_group" json:"security_group"`
	}

	err := mapstructure.Decode(r.Body, &res)

	return res.SecGroup, err
}

// CreateResult represents the result of a create operation.
type CreateResult struct {
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
