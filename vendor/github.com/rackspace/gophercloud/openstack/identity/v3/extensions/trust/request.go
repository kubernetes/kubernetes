package trust

import (
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/openstack"
	token3 "github.com/rackspace/gophercloud/openstack/identity/v3/tokens"
)

type AuthOptionsExt struct {
	token3.AuthOptions
	TrustID string
}

func (ao AuthOptionsExt) ToAuthOptionsV3Map(c *gophercloud.ServiceClient, scope *token3.Scope) (map[string]interface{}, error) {
	//Passing scope value to nil to add scope later in this function.
	authMap, err := ao.AuthOptions.ToAuthOptionsV3Map(c, nil)
	if err != nil {
		return nil, err
	}
	authMap = authMap["auth"].(map[string]interface{})

	// Add a "scope" element if a Scope has been provided.
	if ao.TrustID != "" {
		// TrustID provided.
		authMap["scope"] = map[string]interface{}{
			"OS-TRUST:trust": map[string]interface{}{
				"id": ao.TrustID,
			},
		}
	} else {
		return nil, token3.ErrScopeEmpty
	}
	return map[string]interface{}{"auth": authMap}, nil
}

// AuthenticateV3 explicitly authenticates against the identity v3 service.
func AuthenticateV3Trust(client *gophercloud.ProviderClient, options AuthOptionsExt) error {
	return trustv3auth(client, "", options)
}

func trustv3auth(client *gophercloud.ProviderClient, endpoint string, options AuthOptionsExt) error {
	//In case of Trust TokenId would be Provided so we have to populate the value in service client
	//to not throw password error,also if it is not provided it will be empty which maintains
	//the current implementation.
	client.TokenID = options.AuthOptions.TokenID
	// Override the generated service endpoint with the one returned by the version endpoint.
	v3Client := openstack.NewIdentityV3(client)
	if endpoint != "" {
		v3Client.Endpoint = endpoint
	}

	// copy the auth options to a local variable that we can change. `options`
	// needs to stay as-is for reauth purposes
	v3Options := options

	var scope *token3.Scope

	result := token3.Create(v3Client, v3Options, scope)

	token, err := result.ExtractToken()
	if err != nil {
		return err
	}

	catalog, err := result.ExtractServiceCatalog()
	if err != nil {
		return err
	}

	client.TokenID = token.ID

	if options.AuthOptions.AllowReauth {
		client.ReauthFunc = func() error {
			client.TokenID = ""
			return trustv3auth(client, endpoint, options)
		}
	}
	client.EndpointLocator = func(opts gophercloud.EndpointOpts) (string, error) {
		return openstack.V3EndpointURL(catalog, opts)
	}

	return nil
}
