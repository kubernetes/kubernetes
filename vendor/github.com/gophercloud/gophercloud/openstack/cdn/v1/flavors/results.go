package flavors

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// Provider represents a provider for a particular flavor.
type Provider struct {
	// Specifies the name of the provider. The name must not exceed 64 bytes in
	// length and is limited to unicode, digits, underscores, and hyphens.
	Provider string `json:"provider"`
	// Specifies a list with an href where rel is provider_url.
	Links []gophercloud.Link `json:"links"`
}

// Flavor represents a mapping configuration to a CDN provider.
type Flavor struct {
	// Specifies the name of the flavor. The name must not exceed 64 bytes in
	// length and is limited to unicode, digits, underscores, and hyphens.
	ID string `json:"id"`
	// Specifies the list of providers mapped to this flavor.
	Providers []Provider `json:"providers"`
	// Specifies the self-navigating JSON document paths.
	Links []gophercloud.Link `json:"links"`
}

// FlavorPage is the page returned by a pager when traversing over a
// collection of CDN flavors.
type FlavorPage struct {
	pagination.SinglePageBase
}

// IsEmpty returns true if a FlavorPage contains no Flavors.
func (r FlavorPage) IsEmpty() (bool, error) {
	flavors, err := ExtractFlavors(r)
	return len(flavors) == 0, err
}

// ExtractFlavors extracts and returns Flavors. It is used while iterating over
// a flavors.List call.
func ExtractFlavors(r pagination.Page) ([]Flavor, error) {
	var s struct {
		Flavors []Flavor `json:"flavors"`
	}
	err := (r.(FlavorPage)).ExtractInto(&s)
	return s.Flavors, err
}

// GetResult represents the result of a get operation.
type GetResult struct {
	gophercloud.Result
}

// Extract is a function that extracts a flavor from a GetResult.
func (r GetResult) Extract() (*Flavor, error) {
	var s *Flavor
	err := r.ExtractInto(&s)
	return s, err
}
