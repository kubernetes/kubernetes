package credentials

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// Credential represents the Credential object
type Credential struct {
	// The ID of the credential.
	ID string `json:"id"`
	// Serialized Blob Credential.
	Blob string `json:"blob"`
	// ID of the user who owns the credential.
	UserID string `json:"user_id"`
	// The type of the credential.
	Type string `json:"type"`
	// The ID of the project the credential was created for.
	ProjectID string `json:"project_id"`
	// Links contains referencing links to the credential.
	Links map[string]interface{} `json:"links"`
}

type credentialResult struct {
	gophercloud.Result
}

// GetResult is the response from a Get operation. Call its Extract method
// to interpret it as a Credential.
type GetResult struct {
	credentialResult
}

// CreateResult is the response from a Create operation. Call its Extract method
// to interpret it as a Credential.
type CreateResult struct {
	credentialResult
}

// DeleteResult is the response from a Delete operation. Call its ExtractErr to
// determine if the request succeeded or failed.
type DeleteResult struct {
	gophercloud.ErrResult
}

// UpdateResult is the result of an Update request. Call its Extract method to
// interpret it as a Credential
type UpdateResult struct {
	credentialResult
}

// a CredentialPage is a single page of a Credential results.
type CredentialPage struct {
	pagination.LinkedPageBase
}

// IsEmpty determines whether or not a CredentialPage contains any results.
func (r CredentialPage) IsEmpty() (bool, error) {
	credentials, err := ExtractCredentials(r)
	return len(credentials) == 0, err
}

// NextPageURL extracts the "next" link from the links section of the result.
func (r CredentialPage) NextPageURL() (string, error) {
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

// Extract a Credential returns a slice of Credentials contained in a single page of results.
func ExtractCredentials(r pagination.Page) ([]Credential, error) {
	var s struct {
		Credentials []Credential `json:"credentials"`
	}
	err := (r.(CredentialPage)).ExtractInto(&s)
	return s.Credentials, err
}

// Extract interprets any credential results as a Credential.
func (r credentialResult) Extract() (*Credential, error) {
	var s struct {
		Credential *Credential `json:"credential"`
	}
	err := r.ExtractInto(&s)
	return s.Credential, err
}
