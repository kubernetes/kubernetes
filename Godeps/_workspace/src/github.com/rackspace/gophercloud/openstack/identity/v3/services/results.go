package services

import (
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"

	"github.com/mitchellh/mapstructure"
)

type commonResult struct {
	gophercloud.Result
}

// Extract interprets a GetResult, CreateResult or UpdateResult as a concrete Service.
// An error is returned if the original call or the extraction failed.
func (r commonResult) Extract() (*Service, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var res struct {
		Service `json:"service"`
	}

	err := mapstructure.Decode(r.Body, &res)

	return &res.Service, err
}

// CreateResult is the deferred result of a Create call.
type CreateResult struct {
	commonResult
}

// GetResult is the deferred result of a Get call.
type GetResult struct {
	commonResult
}

// UpdateResult is the deferred result of an Update call.
type UpdateResult struct {
	commonResult
}

// DeleteResult is the deferred result of an Delete call.
type DeleteResult struct {
	gophercloud.ErrResult
}

// Service is the result of a list or information query.
type Service struct {
	Description *string `json:"description,omitempty"`
	ID          string  `json:"id"`
	Name        string  `json:"name"`
	Type        string  `json:"type"`
}

// ServicePage is a single page of Service results.
type ServicePage struct {
	pagination.LinkedPageBase
}

// IsEmpty returns true if the page contains no results.
func (p ServicePage) IsEmpty() (bool, error) {
	services, err := ExtractServices(p)
	if err != nil {
		return true, err
	}
	return len(services) == 0, nil
}

// ExtractServices extracts a slice of Services from a Collection acquired from List.
func ExtractServices(page pagination.Page) ([]Service, error) {
	var response struct {
		Services []Service `mapstructure:"services"`
	}

	err := mapstructure.Decode(page.(ServicePage).Body, &response)
	return response.Services, err
}
