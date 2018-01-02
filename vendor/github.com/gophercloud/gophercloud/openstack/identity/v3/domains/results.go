package domains

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// A Domain is a collection of projects, users, and roles.
type Domain struct {
	// Description is the description of the Domain.
	Description string `json:"description"`

	// Enabled is whether or not the domain is enabled.
	Enabled bool `json:"enabled"`

	// ID is the unique ID of the domain.
	ID string `json:"id"`

	// Links contains referencing links to the domain.
	Links map[string]interface{} `json:"links"`

	// Name is the name of the domain.
	Name string `json:"name"`
}

type domainResult struct {
	gophercloud.Result
}

// GetResult is the response from a Get operation. Call its Extract method
// to interpret it as a Domain.
type GetResult struct {
	domainResult
}

// CreateResult is the response from a Create operation. Call its Extract method
// to interpret it as a Domain.
type CreateResult struct {
	domainResult
}

// DeleteResult is the response from a Delete operation. Call its ExtractErr to
// determine if the request succeeded or failed.
type DeleteResult struct {
	gophercloud.ErrResult
}

// UpdateResult is the result of an Update request. Call its Extract method to
// interpret it as a Domain.
type UpdateResult struct {
	domainResult
}

// DomainPage is a single page of Domain results.
type DomainPage struct {
	pagination.LinkedPageBase
}

// IsEmpty determines whether or not a page of Domains contains any results.
func (r DomainPage) IsEmpty() (bool, error) {
	domains, err := ExtractDomains(r)
	return len(domains) == 0, err
}

// NextPageURL extracts the "next" link from the links section of the result.
func (r DomainPage) NextPageURL() (string, error) {
	var s struct {
		Links struct {
			Next     string `json:"next"`
			Previous string `json:"previous"`
		} `json:"links"`
	}
	err := r.ExtractInto(&s)
	if err != nil {
		return "", err
	}
	return s.Links.Next, err
}

// ExtractDomains returns a slice of Domains contained in a single page of
// results.
func ExtractDomains(r pagination.Page) ([]Domain, error) {
	var s struct {
		Domains []Domain `json:"domains"`
	}
	err := (r.(DomainPage)).ExtractInto(&s)
	return s.Domains, err
}

// Extract interprets any domainResults as a Domain.
func (r domainResult) Extract() (*Domain, error) {
	var s struct {
		Domain *Domain `json:"domain"`
	}
	err := r.ExtractInto(&s)
	return s.Domain, err
}
