package secrets

import (
	"encoding/json"
	"io"
	"io/ioutil"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// Secret represents a secret stored in the key manager service.
type Secret struct {
	// BitLength is the bit length of the secret.
	BitLength int `json:"bit_length"`

	// Algorithm is the algorithm type of the secret.
	Algorithm string `json:"algorithm"`

	// Expiration is the expiration date of the secret.
	Expiration time.Time `json:"-"`

	// ContentTypes are the content types of the secret.
	ContentTypes map[string]string `json:"content_types"`

	// Created is the created date of the secret.
	Created time.Time `json:"-"`

	// CreatorID is the creator of the secret.
	CreatorID string `json:"creator_id"`

	// Mode is the mode of the secret.
	Mode string `json:"mode"`

	// Name is the name of the secret.
	Name string `json:"name"`

	// SecretRef is the URL to the secret.
	SecretRef string `json:"secret_ref"`

	// SecretType represents the type of secret.
	SecretType string `json:"secret_type"`

	// Status represents the status of the secret.
	Status string `json:"status"`

	// Updated is the updated date of the secret.
	Updated time.Time `json:"-"`
}

func (r *Secret) UnmarshalJSON(b []byte) error {
	type tmp Secret
	var s struct {
		tmp
		Created    gophercloud.JSONRFC3339NoZ `json:"created"`
		Updated    gophercloud.JSONRFC3339NoZ `json:"updated"`
		Expiration gophercloud.JSONRFC3339NoZ `json:"expiration"`
	}
	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}
	*r = Secret(s.tmp)

	r.Created = time.Time(s.Created)
	r.Updated = time.Time(s.Updated)
	r.Expiration = time.Time(s.Expiration)

	return nil
}

type commonResult struct {
	gophercloud.Result
}

// Extract interprets any commonResult as a Secret.
func (r commonResult) Extract() (*Secret, error) {
	var s *Secret
	err := r.ExtractInto(&s)
	return s, err
}

// GetResult is the response from a Get operation. Call its Extract method
// to interpret it as a secrets.
type GetResult struct {
	commonResult
}

// CreateResult is the response from a Create operation. Call its Extract method
// to interpret it as a secrets.
type CreateResult struct {
	commonResult
}

// UpdateResult is the response from an Update operation. Call its ExtractErr to
// determine if the request succeeded or failed.
type UpdateResult struct {
	gophercloud.ErrResult
}

// DeleteResult is the response from a Delete operation. Call its ExtractErr to
// determine if the request succeeded or failed.
type DeleteResult struct {
	gophercloud.ErrResult
}

// PayloadResult is the response from a GetPayload operation. Call its Extract
// method to extract the payload as a string.
type PayloadResult struct {
	gophercloud.Result
	Body io.ReadCloser
}

// Extract is a function that takes a PayloadResult's io.Reader body
// and reads all available data into a slice of bytes. Please be aware that due
// to the nature of io.Reader is forward-only - meaning that it can only be read
// once and not rewound. You can recreate a reader from the output of this
// function by using bytes.NewReader(downloadBytes)
func (r PayloadResult) Extract() ([]byte, error) {
	if r.Err != nil {
		return nil, r.Err
	}
	defer r.Body.Close()
	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		return nil, err
	}
	r.Body.Close()
	return body, nil
}

// SecretPage is a single page of secrets results.
type SecretPage struct {
	pagination.LinkedPageBase
}

// IsEmpty determines whether or not a page of secrets contains any results.
func (r SecretPage) IsEmpty() (bool, error) {
	secrets, err := ExtractSecrets(r)
	return len(secrets) == 0, err
}

// NextPageURL extracts the "next" link from the links section of the result.
func (r SecretPage) NextPageURL() (string, error) {
	var s struct {
		Next     string `json:"next"`
		Previous string `json:"previous"`
	}
	err := r.ExtractInto(&s)
	if err != nil {
		return "", err
	}
	return s.Next, err
}

// ExtractSecrets returns a slice of Secrets contained in a single page of
// results.
func ExtractSecrets(r pagination.Page) ([]Secret, error) {
	var s struct {
		Secrets []Secret `json:"secrets"`
	}
	err := (r.(SecretPage)).ExtractInto(&s)
	return s.Secrets, err
}

// MetadataResult is the result of a metadata request. Call its Extract method
// to interpret it as a map[string]string.
type MetadataResult struct {
	gophercloud.Result
}

// Extract interprets any MetadataResult as map[string]string.
func (r MetadataResult) Extract() (map[string]string, error) {
	var s struct {
		Metadata map[string]string `json:"metadata"`
	}
	err := r.ExtractInto(&s)
	return s.Metadata, err
}

// MetadataCreateResult is the result of a metadata create request. Call its
// Extract method to interpret it as a map[string]string.
type MetadataCreateResult struct {
	gophercloud.Result
}

// Extract interprets any MetadataCreateResult as a map[string]string.
func (r MetadataCreateResult) Extract() (map[string]string, error) {
	var s map[string]string
	err := r.ExtractInto(&s)
	return s, err
}

// Metadatum represents an individual metadata.
type Metadatum struct {
	Key   string `json:"key"`
	Value string `json:"value"`
}

// MetadatumResult is the result of a metadatum request. Call its
// Extract method to interpret it as a map[string]string.
type MetadatumResult struct {
	gophercloud.Result
}

// Extract interprets any MetadatumResult as a map[string]string.
func (r MetadatumResult) Extract() (*Metadatum, error) {
	var s *Metadatum
	err := r.ExtractInto(&s)
	return s, err
}

// MetadatumCreateResult is the response from a metadata Create operation. Call
// it's ExtractErr to determine if the request succeeded or failed.
//
// NOTE: This could be a MetadatumResponse but, at the time of testing, it looks
// like Barbican was returning errneous JSON in the response.
type MetadatumCreateResult struct {
	gophercloud.ErrResult
}

// MetadatumDeleteResult is the response from a metadatum Delete operation. Call
// its ExtractErr to determine if the request succeeded or failed.
type MetadatumDeleteResult struct {
	gophercloud.ErrResult
}
