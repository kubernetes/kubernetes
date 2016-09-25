package tokens

import (
	"errors"

	"github.com/rackspace/gophercloud"
	os "github.com/rackspace/gophercloud/openstack/identity/v2/tokens"
)

var (
	// ErrPasswordProvided is returned if both a password and an API key are provided to Create.
	ErrPasswordProvided = errors.New("Please provide either a password or an API key.")
)

// AuthOptions wraps the OpenStack AuthOptions struct to be able to customize the request body
// when API key authentication is used.
type AuthOptions struct {
	os.AuthOptions
}

// WrapOptions embeds a root AuthOptions struct in a package-specific one.
func WrapOptions(original gophercloud.AuthOptions) AuthOptions {
	return AuthOptions{AuthOptions: os.WrapOptions(original)}
}

// ToTokenCreateMap serializes an AuthOptions into a request body. If an API key is provided, it
// will be used, otherwise
func (auth AuthOptions) ToTokenCreateMap() (map[string]interface{}, error) {
	if auth.APIKey == "" {
		return auth.AuthOptions.ToTokenCreateMap()
	}

	// Verify that other required attributes are present.
	if auth.Username == "" {
		return nil, os.ErrUsernameRequired
	}

	authMap := make(map[string]interface{})

	authMap["RAX-KSKEY:apiKeyCredentials"] = map[string]interface{}{
		"username": auth.Username,
		"apiKey":   auth.APIKey,
	}

	if auth.TenantID != "" {
		authMap["tenantId"] = auth.TenantID
	}
	if auth.TenantName != "" {
		authMap["tenantName"] = auth.TenantName
	}

	return map[string]interface{}{"auth": authMap}, nil
}

// Create authenticates to Rackspace's identity service and attempts to acquire a Token. Rather
// than interact with this service directly, users should generally call
// rackspace.AuthenticatedClient().
func Create(client *gophercloud.ServiceClient, auth AuthOptions) os.CreateResult {
	return os.Create(client, auth)
}
