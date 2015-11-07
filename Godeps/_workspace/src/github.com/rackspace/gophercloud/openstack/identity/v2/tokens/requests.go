package tokens

import (
	"fmt"

	"github.com/rackspace/gophercloud"
)

// AuthOptionsBuilder describes any argument that may be passed to the Create call.
type AuthOptionsBuilder interface {

	// ToTokenCreateMap assembles the Create request body, returning an error if parameters are
	// missing or inconsistent.
	ToTokenCreateMap() (map[string]interface{}, error)
}

// AuthOptions wraps a gophercloud AuthOptions in order to adhere to the AuthOptionsBuilder
// interface.
type AuthOptions struct {
	gophercloud.AuthOptions
}

// WrapOptions embeds a root AuthOptions struct in a package-specific one.
func WrapOptions(original gophercloud.AuthOptions) AuthOptions {
	return AuthOptions{AuthOptions: original}
}

// ToTokenCreateMap converts AuthOptions into nested maps that can be serialized into a JSON
// request.
func (auth AuthOptions) ToTokenCreateMap() (map[string]interface{}, error) {
	// Error out if an unsupported auth option is present.
	if auth.UserID != "" {
		return nil, ErrUserIDProvided
	}
	if auth.APIKey != "" {
		return nil, ErrAPIKeyProvided
	}
	if auth.DomainID != "" {
		return nil, ErrDomainIDProvided
	}
	if auth.DomainName != "" {
		return nil, ErrDomainNameProvided
	}

	// Populate the request map.
	authMap := make(map[string]interface{})

	if auth.Username != "" {
		if auth.Password != "" {
			authMap["passwordCredentials"] = map[string]interface{}{
				"username": auth.Username,
				"password": auth.Password,
			}
		} else {
			return nil, ErrPasswordRequired
		}
	} else if auth.TokenID != "" {
		authMap["token"] = map[string]interface{}{
			"id": auth.TokenID,
		}
	} else {
		return nil, fmt.Errorf("You must provide either username/password or tenantID/token values.")
	}

	if auth.TenantID != "" {
		authMap["tenantId"] = auth.TenantID
	}
	if auth.TenantName != "" {
		authMap["tenantName"] = auth.TenantName
	}

	return map[string]interface{}{"auth": authMap}, nil
}

// Create authenticates to the identity service and attempts to acquire a Token.
// If successful, the CreateResult
// Generally, rather than interact with this call directly, end users should call openstack.AuthenticatedClient(),
// which abstracts all of the gory details about navigating service catalogs and such.
func Create(client *gophercloud.ServiceClient, auth AuthOptionsBuilder) CreateResult {
	request, err := auth.ToTokenCreateMap()
	if err != nil {
		return CreateResult{gophercloud.Result{Err: err}}
	}

	var result CreateResult
	_, result.Err = client.Post(CreateURL(client), request, &result.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200, 203},
	})
	return result
}
