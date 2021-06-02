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
	RegionID  string `json:"region_id"`
	Interface string `json:"interface"`
	URL       string `json:"url"`
}

// CatalogEntry provides a type-safe interface to an Identity API V3 service
// catalog listing. Each class of service, such as cloud DNS or block storage
// services, could have multiple CatalogEntry representing it (one by interface
// type, e.g public, admin or internal).
//
// Note: when looking for the desired service, try, whenever possible, to key
// off the type field. Otherwise, you'll tie the representation of the service
// to a specific provider.
type CatalogEntry struct {
	// Service ID
	ID string `json:"id"`

	// Name will contain the provider-specified name for the service.
	Name string `json:"name"`

	// Type will contain a type string if OpenStack defines a type for the
	// service. Otherwise, for provider-specific services, the provider may
	// assign their own type strings.
	Type string `json:"type"`

	// Endpoints will let the caller iterate over all the different endpoints that
	// may exist for the service.
	Endpoints []Endpoint `json:"endpoints"`
}

// ServiceCatalog provides a view into the service catalog from a previous,
// successful authentication.
type ServiceCatalog struct {
	Entries []CatalogEntry `json:"catalog"`
}

// Domain provides information about the domain to which this token grants
// access.
type Domain struct {
	ID   string `json:"id"`
	Name string `json:"name"`
}

// User represents a user resource that exists in the Identity Service.
type User struct {
	Domain Domain `json:"domain"`
	ID     string `json:"id"`
	Name   string `json:"name"`
}

// Role provides information about roles to which User is authorized.
type Role struct {
	ID   string `json:"id"`
	Name string `json:"name"`
}

// Project provides information about project to which User is authorized.
type Project struct {
	Domain Domain `json:"domain"`
	ID     string `json:"id"`
	Name   string `json:"name"`
}

// commonResult is the response from a request. A commonResult has various
// methods which can be used to extract different details about the result.
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

// ExtractTokenID implements the gophercloud.AuthResult interface. The returned
// string is the same as the ID field of the Token struct returned from
// ExtractToken().
func (r CreateResult) ExtractTokenID() (string, error) {
	return r.Header.Get("X-Subject-Token"), r.Err
}

// ExtractServiceCatalog returns the ServiceCatalog that was generated along
// with the user's Token.
func (r commonResult) ExtractServiceCatalog() (*ServiceCatalog, error) {
	var s ServiceCatalog
	err := r.ExtractInto(&s)
	return &s, err
}

// ExtractUser returns the User that is the owner of the Token.
func (r commonResult) ExtractUser() (*User, error) {
	var s struct {
		User *User `json:"user"`
	}
	err := r.ExtractInto(&s)
	return s.User, err
}

// ExtractRoles returns Roles to which User is authorized.
func (r commonResult) ExtractRoles() ([]Role, error) {
	var s struct {
		Roles []Role `json:"roles"`
	}
	err := r.ExtractInto(&s)
	return s.Roles, err
}

// ExtractProject returns Project to which User is authorized.
func (r commonResult) ExtractProject() (*Project, error) {
	var s struct {
		Project *Project `json:"project"`
	}
	err := r.ExtractInto(&s)
	return s.Project, err
}

// CreateResult is the response from a Create request. Use ExtractToken()
// to interpret it as a Token, or ExtractServiceCatalog() to interpret it
// as a service catalog.
type CreateResult struct {
	commonResult
}

// GetResult is the response from a Get request. Use ExtractToken()
// to interpret it as a Token, or ExtractServiceCatalog() to interpret it
// as a service catalog.
type GetResult struct {
	commonResult
}

// RevokeResult is response from a Revoke request.
type RevokeResult struct {
	commonResult
}

// Token is a string that grants a user access to a controlled set of services
// in an OpenStack provider. Each Token is valid for a set length of time.
type Token struct {
	// ID is the issued token.
	ID string `json:"id"`

	// ExpiresAt is the timestamp at which this token will no longer be accepted.
	ExpiresAt time.Time `json:"expires_at"`
}

func (r commonResult) ExtractInto(v interface{}) error {
	return r.ExtractIntoStructPtr(v, "token")
}
