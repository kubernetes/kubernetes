package tokens

import (
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/identity/v2/tenants"
)

// Token provides only the most basic information related to an authentication
// token.
type Token struct {
	// ID provides the primary means of identifying a user to the OpenStack API.
	// OpenStack defines this field as an opaque value, so do not depend on its
	// content. It is safe, however, to compare for equality.
	ID string

	// ExpiresAt provides a timestamp in ISO 8601 format, indicating when the
	// authentication token becomes invalid. After this point in time, future
	// API requests made using this  authentication token will respond with
	// errors. Either the caller will need to reauthenticate manually, or more
	// preferably, the caller should exploit automatic re-authentication.
	// See the AuthOptions structure for more details.
	ExpiresAt time.Time

	// Tenant provides information about the tenant to which this token grants
	// access.
	Tenant tenants.Tenant
}

// Role is a role for a user.
type Role struct {
	Name string `json:"name"`
}

// User is an OpenStack user.
type User struct {
	ID       string `json:"id"`
	Name     string `json:"name"`
	UserName string `json:"username"`
	Roles    []Role `json:"roles"`
}

// Endpoint represents a single API endpoint offered by a service.
// It provides the public and internal URLs, if supported, along with a region
// specifier, again if provided.
//
// The significance of the Region field will depend upon your provider.
//
// In addition, the interface offered by the service will have version
// information associated with it through the VersionId, VersionInfo, and
// VersionList fields, if provided or supported.
//
// In all cases, fields which aren't supported by the provider and service
// combined will assume a zero-value ("").
type Endpoint struct {
	TenantID    string `json:"tenantId"`
	PublicURL   string `json:"publicURL"`
	InternalURL string `json:"internalURL"`
	AdminURL    string `json:"adminURL"`
	Region      string `json:"region"`
	VersionID   string `json:"versionId"`
	VersionInfo string `json:"versionInfo"`
	VersionList string `json:"versionList"`
}

// CatalogEntry provides a type-safe interface to an Identity API V2 service
// catalog listing.
//
// Each class of service, such as cloud DNS or block storage services, will have
// a single CatalogEntry representing it.
//
// Note: when looking for the desired service, try, whenever possible, to key
// off the type field. Otherwise, you'll tie the representation of the service
// to a specific provider.
type CatalogEntry struct {
	// Name will contain the provider-specified name for the service.
	Name string `json:"name"`

	// Type will contain a type string if OpenStack defines a type for the
	// service. Otherwise, for provider-specific services, the provider may assign
	// their own type strings.
	Type string `json:"type"`

	// Endpoints will let the caller iterate over all the different endpoints that
	// may exist for the service.
	Endpoints []Endpoint `json:"endpoints"`
}

// ServiceCatalog provides a view into the service catalog from a previous,
// successful authentication.
type ServiceCatalog struct {
	Entries []CatalogEntry
}

// CreateResult is the response from a Create request. Use ExtractToken() to
// interpret it as a Token, or ExtractServiceCatalog() to interpret it as a
// service catalog.
type CreateResult struct {
	gophercloud.Result
}

// GetResult is the deferred response from a Get call, which is the same with a
// Created token. Use ExtractUser() to interpret it as a User.
type GetResult struct {
	CreateResult
}

// ExtractToken returns the just-created Token from a CreateResult.
func (r CreateResult) ExtractToken() (*Token, error) {
	var s struct {
		Access struct {
			Token struct {
				Expires string         `json:"expires"`
				ID      string         `json:"id"`
				Tenant  tenants.Tenant `json:"tenant"`
			} `json:"token"`
		} `json:"access"`
	}

	err := r.ExtractInto(&s)
	if err != nil {
		return nil, err
	}

	expiresTs, err := time.Parse(gophercloud.RFC3339Milli, s.Access.Token.Expires)
	if err != nil {
		return nil, err
	}

	return &Token{
		ID:        s.Access.Token.ID,
		ExpiresAt: expiresTs,
		Tenant:    s.Access.Token.Tenant,
	}, nil
}

// ExtractServiceCatalog returns the ServiceCatalog that was generated along
// with the user's Token.
func (r CreateResult) ExtractServiceCatalog() (*ServiceCatalog, error) {
	var s struct {
		Access struct {
			Entries []CatalogEntry `json:"serviceCatalog"`
		} `json:"access"`
	}
	err := r.ExtractInto(&s)
	return &ServiceCatalog{Entries: s.Access.Entries}, err
}

// ExtractUser returns the User from a GetResult.
func (r GetResult) ExtractUser() (*User, error) {
	var s struct {
		Access struct {
			User User `json:"user"`
		} `json:"access"`
	}
	err := r.ExtractInto(&s)
	return &s.Access.User, err
}
