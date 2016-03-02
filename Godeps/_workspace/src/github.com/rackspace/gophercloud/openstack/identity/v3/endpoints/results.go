package endpoints

import (
	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

type commonResult struct {
	gophercloud.Result
}

// Extract interprets a GetResult, CreateResult or UpdateResult as a concrete Endpoint.
// An error is returned if the original call or the extraction failed.
func (r commonResult) Extract() (*Endpoint, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var res struct {
		Endpoint `json:"endpoint"`
	}

	err := mapstructure.Decode(r.Body, &res)

	return &res.Endpoint, err
}

// CreateResult is the deferred result of a Create call.
type CreateResult struct {
	commonResult
}

// createErr quickly wraps an error in a CreateResult.
func createErr(err error) CreateResult {
	return CreateResult{commonResult{gophercloud.Result{Err: err}}}
}

// UpdateResult is the deferred result of an Update call.
type UpdateResult struct {
	commonResult
}

// DeleteResult is the deferred result of an Delete call.
type DeleteResult struct {
	gophercloud.ErrResult
}

// Endpoint describes the entry point for another service's API.
type Endpoint struct {
	ID           string                   `mapstructure:"id" json:"id"`
	Availability gophercloud.Availability `mapstructure:"interface" json:"interface"`
	Name         string                   `mapstructure:"name" json:"name"`
	Region       string                   `mapstructure:"region" json:"region"`
	ServiceID    string                   `mapstructure:"service_id" json:"service_id"`
	URL          string                   `mapstructure:"url" json:"url"`
}

// EndpointPage is a single page of Endpoint results.
type EndpointPage struct {
	pagination.LinkedPageBase
}

// IsEmpty returns true if no Endpoints were returned.
func (p EndpointPage) IsEmpty() (bool, error) {
	es, err := ExtractEndpoints(p)
	if err != nil {
		return true, err
	}
	return len(es) == 0, nil
}

// ExtractEndpoints extracts an Endpoint slice from a Page.
func ExtractEndpoints(page pagination.Page) ([]Endpoint, error) {
	var response struct {
		Endpoints []Endpoint `mapstructure:"endpoints"`
	}

	err := mapstructure.Decode(page.(EndpointPage).Body, &response)

	return response.Endpoints, err
}
