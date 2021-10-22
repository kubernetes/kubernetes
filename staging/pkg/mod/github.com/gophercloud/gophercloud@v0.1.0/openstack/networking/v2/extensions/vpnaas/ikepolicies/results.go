package ikepolicies

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// Policy is an IKE Policy
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

	// EncryptionAlgorithm is the encryption algorithm
	EncryptionAlgorithm string `json:"encryption_algorithm"`

	// PFS is the Perfect forward secrecy (PFS) mode
	PFS string `json:"pfs"`

	// Lifetime is the lifetime of the security association
	Lifetime Lifetime `json:"lifetime"`

	// ID is the ID of the policy
	ID string `json:"id"`

	// Phase1NegotiationMode is the IKE mode
	Phase1NegotiationMode string `json:"phase1_negotiation_mode"`

	// IKEVersion is the IKE version.
	IKEVersion string `json:"ike_version"`
}

type commonResult struct {
	gophercloud.Result
}
type Lifetime struct {
	// Units is the unit for the lifetime
	// Default is seconds
	Units string `json:"units"`

	// Value is the lifetime
	// Default is 3600
	Value int `json:"value"`
}

// Extract is a function that accepts a result and extracts an IKE Policy.
func (r commonResult) Extract() (*Policy, error) {
	var s struct {
		Policy *Policy `json:"ikepolicy"`
	}
	err := r.ExtractInto(&s)
	return s.Policy, err
}

// PolicyPage is the page returned by a pager when traversing over a
// collection of Policies.
type PolicyPage struct {
	pagination.LinkedPageBase
}

// NextPageURL is invoked when a paginated collection of IKE policies has
// reached the end of a page and the pager seeks to traverse over a new one.
// In order to do this, it needs to construct the next page's URL.
func (r PolicyPage) NextPageURL() (string, error) {
	var s struct {
		Links []gophercloud.Link `json:"ikepolicies_links"`
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
		Policies []Policy `json:"ikepolicies"`
	}
	err := (r.(PolicyPage)).ExtractInto(&s)
	return s.Policies, err
}

// CreateResult represents the result of a Create operation. Call its Extract method to
// interpret it as a Policy.
type CreateResult struct {
	commonResult
}

// GetResult represents the result of a Get operation. Call its Extract method to
// interpret it as a Policy.
type GetResult struct {
	commonResult
}

// DeleteResult represents the results of a Delete operation. Call its ExtractErr method
// to determine whether the operation succeeded or failed.
type DeleteResult struct {
	gophercloud.ErrResult
}

// UpdateResult represents the result of an update operation. Call its Extract method
// to interpret it as a Policy.
type UpdateResult struct {
	commonResult
}
