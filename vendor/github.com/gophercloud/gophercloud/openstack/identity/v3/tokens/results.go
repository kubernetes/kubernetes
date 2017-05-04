package tokens

import (
	"time"

	"github.com/gophercloud/gophercloud"
)

// Endpoint represents a single API endpoint offered by a service.
// It matches either a public, internal or admin URL.
// If supported, it contains a region specifier, again if provided.
// The significance of the Region field will depend upon your provider.
type Endpoint struct {
	ID        string `json:"id"`
	Region    string `json:"region"`
	Interface string `json:"interface"`
	URL       string `json:"url"`
}

// CatalogEntry provides a type-safe interface to an Identity API V3 service catalog listing.
// Each class of service, such as cloud DNS or block storage services, could have multiple
// CatalogEntry representing it (one by interface type, e.g public, admin or internal).
//
// Note: when looking for the desired service, try, whenever possible, to key off the type field.
// Otherwise, you'll tie the representation of the service to a specific provider.
type CatalogEntry struct {
	// Service ID
	ID string `json:"id"`
	// Name will contain the provider-specified name for the service.
	Name string `json:"name"`
	// Type will contain a type string if OpenStack defines a type for the service.
	// Otherwise, for provider-specific services, the provider may assign their own type strings.
	Type string `json:"type"`
	// Endpoints will let the caller iterate over all the different endpoints that may exist for
	// the service.
	Endpoints []Endpoint `json:"endpoints"`
}

// ServiceCatalog provides a view into the service catalog from a previous, successful authentication.
type ServiceCatalog struct {
	Entries []CatalogEntry `json:"catalog"`
}

// commonResult is the deferred result of a Create or a Get call.
type commonResult struct {
	gophercloud.Result
}

// Extract is a shortcut for ExtractToken.
// This function is deprecated and still present for backward compatibility.
func (r commonResult) Extract() (*Token, error) {
	return r.ExtractToken()
}

// ExtractToken interprets a commonResult as a Token.
func (r commonResult) ExtractToken() (*Token, error) {
	var s Token
	err := r.ExtractInto(&s)
	if err != nil {
		return nil, err
	}

	// Parse the token itself from the stored headers.
	s.ID = r.Header.Get("X-Subject-Token")

	return &s, err
}

// ExtractServiceCatalog returns the ServiceCatalog that was generated along with the user's Token.
func (r CreateResult) ExtractServiceCatalog() (*ServiceCatalog, error) {
	var s ServiceCatalog
	err := r.ExtractInto(&s)
	return &s, err
}

// CreateResult defers the interpretation of a created token.
// Use ExtractToken() to interpret it as a Token, or ExtractServiceCatalog() to interpret it as a service catalog.
type CreateResult struct {
	commonResult
}

// GetResult is the deferred response from a Get call.
type GetResult struct {
	commonResult
}

// RevokeResult is the deferred response from a Revoke call.
type RevokeResult struct {
	commonResult
}

// Token is a string that grants a user access to a controlled set of services in an OpenStack provider.
// Each Token is valid for a set length of time.
type Token struct {
	// ID is the issued token.
	ID string `json:"id"`
	// ExpiresAt is the timestamp at which this token will no longer be accepted.
	ExpiresAt time.Time `json:"expires_at"`
}

func (r commonResult) ExtractInto(v interface{}) error {
	return r.ExtractIntoStructPtr(v, "token")
}
