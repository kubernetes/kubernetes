package tokens

import "github.com/gophercloud/gophercloud"

type PasswordCredentialsV2 struct {
	Username string `json:"username" required:"true"`
	Password string `json:"password" required:"true"`
}

type TokenCredentialsV2 struct {
	ID string `json:"id,omitempty" required:"true"`
}

// AuthOptionsV2 wraps a gophercloud AuthOptions in order to adhere to the AuthOptionsBuilder
// interface.
type AuthOptionsV2 struct {
	PasswordCredentials *PasswordCredentialsV2 `json:"passwordCredentials,omitempty" xor:"TokenCredentials"`

	// The TenantID and TenantName fields are optional for the Identity V2 API.
	// Some providers allow you to specify a TenantName instead of the TenantId.
	// Some require both. Your provider's authentication policies will determine
	// how these fields influence authentication.
	TenantID   string `json:"tenantId,omitempty"`
	TenantName string `json:"tenantName,omitempty"`

	// TokenCredentials allows users to authenticate (possibly as another user) with an
	// authentication token ID.
	TokenCredentials *TokenCredentialsV2 `json:"token,omitempty" xor:"PasswordCredentials"`
}

// AuthOptionsBuilder describes any argument that may be passed to the Create call.
type AuthOptionsBuilder interface {
	// ToTokenCreateMap assembles the Create request body, returning an error if parameters are
	// missing or inconsistent.
	ToTokenV2CreateMap() (map[string]interface{}, error)
}

// AuthOptions are the valid options for Openstack Identity v2 authentication.
// For field descriptions, see gophercloud.AuthOptions.
type AuthOptions struct {
	IdentityEndpoint string `json:"-"`
	Username         string `json:"username,omitempty"`
	Password         string `json:"password,omitempty"`
	TenantID         string `json:"tenantId,omitempty"`
	TenantName       string `json:"tenantName,omitempty"`
	AllowReauth      bool   `json:"-"`
	TokenID          string
}

// ToTokenV2CreateMap allows AuthOptions to satisfy the AuthOptionsBuilder
// interface in the v2 tokens package
func (opts AuthOptions) ToTokenV2CreateMap() (map[string]interface{}, error) {
	v2Opts := AuthOptionsV2{
		TenantID:   opts.TenantID,
		TenantName: opts.TenantName,
	}

	if opts.Password != "" {
		v2Opts.PasswordCredentials = &PasswordCredentialsV2{
			Username: opts.Username,
			Password: opts.Password,
		}
	} else {
		v2Opts.TokenCredentials = &TokenCredentialsV2{
			ID: opts.TokenID,
		}
	}

	b, err := gophercloud.BuildRequestBody(v2Opts, "auth")
	if err != nil {
		return nil, err
	}
	return b, nil
}

// Create authenticates to the identity service and attempts to acquire a Token.
// If successful, the CreateResult
// Generally, rather than interact with this call directly, end users should call openstack.AuthenticatedClient(),
// which abstracts all of the gory details about navigating service catalogs and such.
func Create(client *gophercloud.ServiceClient, auth AuthOptionsBuilder) (r CreateResult) {
	b, err := auth.ToTokenV2CreateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Post(CreateURL(client), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes:     []int{200, 203},
		MoreHeaders: map[string]string{"X-Auth-Token": ""},
	})
	return
}

// Get validates and retrieves information for user's token.
func Get(client *gophercloud.ServiceClient, token string) (r GetResult) {
	_, r.Err = client.Get(GetURL(client, token), &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200, 203},
	})
	return
}
