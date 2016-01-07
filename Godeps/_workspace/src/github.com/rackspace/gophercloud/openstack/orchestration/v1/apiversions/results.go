package apiversions

import (
	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// APIVersion represents an API version for Neutron. It contains the status of
// the API, and its unique ID.
type APIVersion struct {
	Status string             `mapstructure:"status"`
	ID     string             `mapstructure:"id"`
	Links  []gophercloud.Link `mapstructure:"links"`
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
