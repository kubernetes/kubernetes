package apiversions

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// APIVersion represents an API version for Cinder.
type APIVersion struct {
	ID      string `json:"id"`      // unique identifier
	Status  string `json:"status"`  // current status
	Updated string `json:"updated"` // date last updated
}

// APIVersionPage is the page returned by a pager when traversing over a
// collection of API versions.
type APIVersionPage struct {
	pagination.SinglePageBase
}

// IsEmpty checks whether an APIVersionPage struct is empty.
func (r APIVersionPage) IsEmpty() (bool, error) {
	is, err := ExtractAPIVersions(r)
	return len(is) == 0, err
}

// ExtractAPIVersions takes a collection page, extracts all of the elements,
// and returns them a slice of APIVersion structs. It is effectively a cast.
func ExtractAPIVersions(r pagination.Page) ([]APIVersion, error) {
	var s struct {
		Versions []APIVersion `json:"versions"`
	}
	err := (r.(APIVersionPage)).ExtractInto(&s)
	return s.Versions, err
}

// GetResult represents the result of a get operation.
type GetResult struct {
	gophercloud.Result
}

// Extract is a function that accepts a result and extracts an API version resource.
func (r GetResult) Extract() (*APIVersion, error) {
	var s struct {
		Version *APIVersion `json:"version"`
	}
	err := r.ExtractInto(&s)
	return s.Version, err
}
