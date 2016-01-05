package apiversions

import (
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"

	"github.com/mitchellh/mapstructure"
)

// APIVersion represents an API version for Cinder.
type APIVersion struct {
	ID      string `json:"id" mapstructure:"id"`           // unique identifier
	Status  string `json:"status" mapstructure:"status"`   // current status
	Updated string `json:"updated" mapstructure:"updated"` // date last updated
}

// APIVersionPage is the page returned by a pager when traversing over a
// collection of API versions.
type APIVersionPage struct {
	pagination.SinglePageBase
}

// IsEmpty checks whether an APIVersionPage struct is empty.
func (r APIVersionPage) IsEmpty() (bool, error) {
	is, err := ExtractAPIVersions(r)
	if err != nil {
		return true, err
	}
	return len(is) == 0, nil
}

// ExtractAPIVersions takes a collection page, extracts all of the elements,
// and returns them a slice of APIVersion structs. It is effectively a cast.
func ExtractAPIVersions(page pagination.Page) ([]APIVersion, error) {
	var resp struct {
		Versions []APIVersion `mapstructure:"versions"`
	}

	err := mapstructure.Decode(page.(APIVersionPage).Body, &resp)

	return resp.Versions, err
}

// GetResult represents the result of a get operation.
type GetResult struct {
	gophercloud.Result
}

// Extract is a function that accepts a result and extracts an API version resource.
func (r GetResult) Extract() (*APIVersion, error) {
	var resp struct {
		Version *APIVersion `mapstructure:"version"`
	}

	err := mapstructure.Decode(r.Body, &resp)

	return resp.Version, err
}
