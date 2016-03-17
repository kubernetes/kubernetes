package tokens

import (
	"time"

	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/openstack/identity/v2/tenants"
)

// Token provides only the most basic information related to an authentication token.
type Token struct {
	// ID provides the primary means of identifying a user to the OpenStack API.
	// OpenStack defines this field as an opaque value, so do not depend on its content.
	// It is safe, however, to compare for equality.
	ID string

	// ExpiresAt provides a timestamp in ISO 8601 format, indicating when the authentication token becomes invalid.
	// After this point in time, future API requests made using this authentication token will respond with errors.
	// Either the caller will need to reauthenticate manually, or more preferably, the caller should exploit automatic re-authentication.
	// See the AuthOptions structure for more details.
	ExpiresAt time.Time

	// Tenant provides information about the tenant to which this token grants access.
	Tenant tenants.Tenant
}

// Authorization need user info which can get from token authentication's response
type Role struct {
	Name string `mapstructure:"name"`
}
type User struct {
	ID       string `mapstructure:"id"`
	Name     string `mapstructure:"name"`
	UserName string `mapstructure:"username"`
	Roles    []Role `mapstructure:"roles"`
}

// Endpoint represents a single API endpoint offered by a service.
// It provides the public and internal URLs, if supported, along with a region specifier, again if provided.
// The significance of the Region field will depend upon your provider.
//
// In addition, the interface offered by the service will have version information associated with it
// through the VersionId, VersionInfo, and VersionList fields, if provided or supported.
//
// In all cases, fields which aren't supported by the provider and service combined will assume a zero-value ("").
type Endpoint struct {
	TenantID    string `mapstructure:"tenantId"`
	PublicURL   string `mapstructure:"publicURL"`
	InternalURL string `mapstructure:"internalURL"`
	AdminURL    string `mapstructure:"adminURL"`
	Region      string `mapstructure:"region"`
	VersionID   string `mapstructure:"versionId"`
	VersionInfo string `mapstructure:"versionInfo"`
	VersionList string `mapstructure:"versionList"`
}

// CatalogEntry provides a type-safe interface to an Identity API V2 service catalog listing.
// Each class of service, such as cloud DNS or block storage services, will have a single
// CatalogEntry representing it.
//
// Note: when looking for the desired service, try, whenever possible, to key off the type field.
// Otherwise, you'll tie the representation of the service to a specific provider.
type CatalogEntry struct {
	// Name will contain the provider-specified name for the service.
	Name string `mapstructure:"name"`

	// Type will contain a type string if OpenStack defines a type for the service.
	// Otherwise, for provider-specific services, the provider may assign their own type strings.
	Type string `mapstructure:"type"`

	// Endpoints will let the caller iterate over all the different endpoints that may exist for
	// the service.
	Endpoints []Endpoint `mapstructure:"endpoints"`
}

// ServiceCatalog provides a view into the service catalog from a previous, successful authentication.
type ServiceCatalog struct {
	Entries []CatalogEntry
}

// CreateResult defers the interpretation of a created token.
// Use ExtractToken() to interpret it as a Token, or ExtractServiceCatalog() to interpret it as a service catalog.
type CreateResult struct {
	gophercloud.Result
}

// GetResult is the deferred response from a Get call, which is the same with a Created token.
// Use ExtractUser() to interpret it as a User.
type GetResult struct {
	CreateResult
}

// ExtractToken returns the just-created Token from a CreateResult.
func (result CreateResult) ExtractToken() (*Token, error) {
	if result.Err != nil {
		return nil, result.Err
	}

	var response struct {
		Access struct {
			Token struct {
				Expires string         `mapstructure:"expires"`
				ID      string         `mapstructure:"id"`
				Tenant  tenants.Tenant `mapstructure:"tenant"`
			} `mapstructure:"token"`
		} `mapstructure:"access"`
	}

	err := mapstructure.Decode(result.Body, &response)
	if err != nil {
		return nil, err
	}

	expiresTs, err := time.Parse(gophercloud.RFC3339Milli, response.Access.Token.Expires)
	if err != nil {
		return nil, err
	}

	return &Token{
		ID:        response.Access.Token.ID,
		ExpiresAt: expiresTs,
		Tenant:    response.Access.Token.Tenant,
	}, nil
}

// ExtractServiceCatalog returns the ServiceCatalog that was generated along with the user's Token.
func (result CreateResult) ExtractServiceCatalog() (*ServiceCatalog, error) {
	if result.Err != nil {
		return nil, result.Err
	}

	var response struct {
		Access struct {
			Entries []CatalogEntry `mapstructure:"serviceCatalog"`
		} `mapstructure:"access"`
	}

	err := mapstructure.Decode(result.Body, &response)
	if err != nil {
		return nil, err
	}

	return &ServiceCatalog{Entries: response.Access.Entries}, nil
}

// createErr quickly packs an error in a CreateResult.
func createErr(err error) CreateResult {
	return CreateResult{gophercloud.Result{Err: err}}
}

// ExtractUser returns the User from a GetResult.
func (result GetResult) ExtractUser() (*User, error) {
	if result.Err != nil {
		return nil, result.Err
	}

	var response struct {
		Access struct {
			User User `mapstructure:"user"`
		} `mapstructure:"access"`
	}

	err := mapstructure.Decode(result.Body, &response)
	if err != nil {
		return nil, err
	}

	return &response.Access.User, nil
}
