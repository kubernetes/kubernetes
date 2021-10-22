package apiversions

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// APIVersion represents an API version for the Container Infra service.
type APIVersion struct {
	// ID is the unique identifier of the API version.
	ID string `json:"id"`

	// MinVersion is the minimum microversion supported.
	MinVersion string `json:"min_version"`

	// Status is the API versions status.
	Status string `json:"status"`

	// Version is the maximum microversion supported.
	Version string `json:"max_version"`
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
		Versions []APIVersion `json:"versions"`
	}
	err := r.ExtractInto(&s)
	if err != nil {
		return nil, err
	}

	switch len(s.Versions) {
	case 0:
		return nil, ErrVersionNotFound{}
	case 1:
		return &s.Versions[0], nil
	default:
		return nil, ErrMultipleVersionsFound{Count: len(s.Versions)}
	}
}
