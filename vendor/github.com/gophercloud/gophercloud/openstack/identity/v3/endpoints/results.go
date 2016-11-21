package endpoints

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

type commonResult struct {
	gophercloud.Result
}

// Extract interprets a GetResult, CreateResult or UpdateResult as a concrete Endpoint.
// An error is returned if the original call or the extraction failed.
func (r commonResult) Extract() (*Endpoint, error) {
	var s struct {
		Endpoint *Endpoint `json:"endpoint"`
	}
	err := r.ExtractInto(&s)
	return s.Endpoint, err
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
	ID           string                   `json:"id"`
	Availability gophercloud.Availability `json:"interface"`
	Name         string                   `json:"name"`
	Region       string                   `json:"region"`
	ServiceID    string                   `json:"service_id"`
	URL          string                   `json:"url"`
}

// EndpointPage is a single page of Endpoint results.
type EndpointPage struct {
	pagination.LinkedPageBase
}

// IsEmpty returns true if no Endpoints were returned.
func (r EndpointPage) IsEmpty() (bool, error) {
	es, err := ExtractEndpoints(r)
	return len(es) == 0, err
}

// ExtractEndpoints extracts an Endpoint slice from a Page.
func ExtractEndpoints(r pagination.Page) ([]Endpoint, error) {
	var s struct {
		Endpoints []Endpoint `json:"endpoints"`
	}
	err := (r.(EndpointPage)).ExtractInto(&s)
	return s.Endpoints, err
}
