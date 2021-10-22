package addressscopes

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

type commonResult struct {
	gophercloud.Result
}

// Extract is a function that accepts a result and extracts an address-scope resource.
func (r commonResult) Extract() (*AddressScope, error) {
	var s struct {
		AddressScope *AddressScope `json:"address_scope"`
	}
	err := r.ExtractInto(&s)
	return s.AddressScope, err
}

// GetResult represents the result of a get operation. Call its Extract
// method to interpret it as a SubnetPool.
type GetResult struct {
	commonResult
}

// CreateResult represents the result of a create operation. Call its Extract
// method to interpret it as a SubnetPool.
type CreateResult struct {
	commonResult
}

// UpdateResult represents the result of an update operation. Call its Extract
// method to interpret it as an AddressScope.
type UpdateResult struct {
	commonResult
}

// DeleteResult represents the result of a delete operation. Call its
// ExtractErr method to determine if the request succeeded or failed.
type DeleteResult struct {
	gophercloud.ErrResult
}

// AddressScope represents a Neutron address-scope.
type AddressScope struct {
	// ID is the id of the address-scope.
	ID string `json:"id"`

	// Name is the human-readable name of the address-scope.
	Name string `json:"name"`

	// TenantID is the id of the Identity project.
	TenantID string `json:"tenant_id"`

	// ProjectID is the id of the Identity project.
	ProjectID string `json:"project_id"`

	// IPVersion is the IP protocol version.
	IPVersion int `json:"ip_version"`

	// Shared indicates whether this address-scope is shared across all projects.
	Shared bool `json:"shared"`
}

// AddressScopePage stores a single page of AddressScopes from a List() API call.
type AddressScopePage struct {
	pagination.LinkedPageBase
}

// NextPageURL is invoked when a paginated collection of address-scope has
// reached the end of a page and the pager seeks to traverse over a new one.
// In order to do this, it needs to construct the next page's URL.
func (r AddressScopePage) NextPageURL() (string, error) {
	var s struct {
		Links []gophercloud.Link `json:"address_scopes_links"`
	}
	err := r.ExtractInto(&s)
	if err != nil {
		return "", err
	}
	return gophercloud.ExtractNextURL(s.Links)
}

// IsEmpty determines whether or not a AddressScopePage is empty.
func (r AddressScopePage) IsEmpty() (bool, error) {
	addressScopes, err := ExtractAddressScopes(r)
	return len(addressScopes) == 0, err
}

// ExtractAddressScopes interprets the results of a single page from a List()
// API call, producing a slice of AddressScopes structs.
func ExtractAddressScopes(r pagination.Page) ([]AddressScope, error) {
	var s struct {
		AddressScopes []AddressScope `json:"address_scopes"`
	}
	err := (r.(AddressScopePage)).ExtractInto(&s)
	return s.AddressScopes, err
}
