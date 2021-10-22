package ipsecpolicies

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// Policy is an IPSec Policy
type Policy struct {
	// TenantID is the ID of the project
	TenantID string `json:"tenant_id"`

	// ProjectID is the ID of the project
	ProjectID string `json:"project_id"`

	// Description is the human readable description of the policy
	Description string `json:"description"`

	// Name is the human readable name of the policy
	Name string `json:"name"`

	// AuthAlgorithm is the authentication hash algorithm
	AuthAlgorithm string `json:"auth_algorithm"`

	// EncapsulationMode is the encapsulation mode
	EncapsulationMode string `json:"encapsulation_mode"`

	// EncryptionAlgorithm is the encryption algorithm
	EncryptionAlgorithm string `json:"encryption_algorithm"`

	// PFS is the Perfect forward secrecy (PFS) mode
	PFS string `json:"pfs"`

	// TransformProtocol is the transform protocol
	TransformProtocol string `json:"transform_protocol"`

	// Lifetime is the lifetime of the security association
	Lifetime Lifetime `json:"lifetime"`

	// ID is the ID of the policy
	ID string `json:"id"`
}

type Lifetime struct {
	// Units is the unit for the lifetime
	// Default is seconds
	Units string `json:"units"`

	// Value is the lifetime
	// Default is 3600
	Value int `json:"value"`
}

type commonResult struct {
	gophercloud.Result
}

// Extract is a function that accepts a result and extracts an IPSec Policy.
func (r commonResult) Extract() (*Policy, error) {
	var s struct {
		Policy *Policy `json:"ipsecpolicy"`
	}
	err := r.ExtractInto(&s)
	return s.Policy, err
}

// CreateResult represents the result of a create operation. Call its Extract
// method to interpret it as a Policy.
type CreateResult struct {
	commonResult
}

// CreateResult represents the result of a delete operation. Call its ExtractErr method
// to determine if the operation succeeded or failed.
type DeleteResult struct {
	gophercloud.ErrResult
}

// GetResult represents the result of a get operation. Call its Extract
// method to interpret it as a Policy.
type GetResult struct {
	commonResult
}

// PolicyPage is the page returned by a pager when traversing over a
// collection of Policies.
type PolicyPage struct {
	pagination.LinkedPageBase
}

// NextPageURL is invoked when a paginated collection of IPSec policies has
// reached the end of a page and the pager seeks to traverse over a new one.
// In order to do this, it needs to construct the next page's URL.
func (r PolicyPage) NextPageURL() (string, error) {
	var s struct {
		Links []gophercloud.Link `json:"ipsecpolicies_links"`
	}
	err := r.ExtractInto(&s)
	if err != nil {
		return "", err
	}
	return gophercloud.ExtractNextURL(s.Links)
}

// IsEmpty checks whether a PolicyPage struct is empty.
func (r PolicyPage) IsEmpty() (bool, error) {
	is, err := ExtractPolicies(r)
	return len(is) == 0, err
}

// ExtractPolicies accepts a Page struct, specifically a Policy struct,
// and extracts the elements into a slice of Policy structs. In other words,
// a generic collection is mapped into a relevant slice.
func ExtractPolicies(r pagination.Page) ([]Policy, error) {
	var s struct {
		Policies []Policy `json:"ipsecpolicies"`
	}
	err := (r.(PolicyPage)).ExtractInto(&s)
	return s.Policies, err
}

// UpdateResult represents the result of an update operation. Call its Extract
// method to interpret it as a Policy.
type UpdateResult struct {
	commonResult
}
