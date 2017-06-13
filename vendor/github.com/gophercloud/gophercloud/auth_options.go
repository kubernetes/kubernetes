package gophercloud

/*
AuthOptions stores information needed to authenticate to an OpenStack Cloud.
You can populate one manually, or use a provider's AuthOptionsFromEnv() function
to read relevant information from the standard environment variables. Pass one
to a provider's AuthenticatedClient function to authenticate and obtain a
ProviderClient representing an active session on that provider.

Its fields are the union of those recognized by each identity implementation and
provider.
*/
type AuthOptions struct {
	// IdentityEndpoint specifies the HTTP endpoint that is required to work with
	// the Identity API of the appropriate version. While it's ultimately needed by
	// all of the identity services, it will often be populated by a provider-level
	// function.
	IdentityEndpoint string `json:"-"`

	// Username is required if using Identity V2 API. Consult with your provider's
	// control panel to discover your account's username. In Identity V3, either
	// UserID or a combination of Username and DomainID or DomainName are needed.
	Username string `json:"username,omitempty"`
	UserID   string `json:"id,omitempty"`

	Password string `json:"password,omitempty"`

	// At most one of DomainID and DomainName must be provided if using Username
	// with Identity V3. Otherwise, either are optional.
	DomainID   string `json:"id,omitempty"`
	DomainName string `json:"name,omitempty"`

	// The TenantID and TenantName fields are optional for the Identity V2 API.
	// The same fields are known as project_id and project_name in the Identity
	// V3 API, but are collected as TenantID and TenantName here in both cases.
	// Some providers allow you to specify a TenantName instead of the TenantId.
	// Some require both. Your provider's authentication policies will determine
	// how these fields influence authentication.
	// If DomainID or DomainName are provided, they will also apply to TenantName.
	// It is not currently possible to authenticate with Username and a Domain
	// and scope to a Project in a different Domain by using TenantName. To
	// accomplish that, the ProjectID will need to be provided to the TenantID
	// option.
	TenantID   string `json:"tenantId,omitempty"`
	TenantName string `json:"tenantName,omitempty"`

	// AllowReauth should be set to true if you grant permission for Gophercloud to
	// cache your credentials in memory, and to allow Gophercloud to attempt to
	// re-authenticate automatically if/when your token expires.  If you set it to
	// false, it will not cache these settings, but re-authentication will not be
	// possible.  This setting defaults to false.
	//
	// NOTE: The reauth function will try to re-authenticate endlessly if left unchecked.
	// The way to limit the number of attempts is to provide a custom HTTP client to the provider client
	// and provide a transport that implements the RoundTripper interface and stores the number of failed retries.
	// For an example of this, see here: https://github.com/rackspace/rack/blob/1.0.0/auth/clients.go#L311
	AllowReauth bool `json:"-"`

	// TokenID allows users to authenticate (possibly as another user) with an
	// authentication token ID.
	TokenID string `json:"-"`
}

// ToTokenV2CreateMap allows AuthOptions to satisfy the AuthOptionsBuilder
// interface in the v2 tokens package
func (opts AuthOptions) ToTokenV2CreateMap() (map[string]interface{}, error) {
	// Populate the request map.
	authMap := make(map[string]interface{})

	if opts.Username != "" {
		if opts.Password != "" {
			authMap["passwordCredentials"] = map[string]interface{}{
				"username": opts.Username,
				"password": opts.Password,
			}
		} else {
			return nil, ErrMissingInput{Argument: "Password"}
		}
	} else if opts.TokenID != "" {
		authMap["token"] = map[string]interface{}{
			"id": opts.TokenID,
		}
	} else {
		return nil, ErrMissingInput{Argument: "Username"}
	}

	if opts.TenantID != "" {
		authMap["tenantId"] = opts.TenantID
	}
	if opts.TenantName != "" {
		authMap["tenantName"] = opts.TenantName
	}

	return map[string]interface{}{"auth": authMap}, nil
}

func (opts *AuthOptions) ToTokenV3CreateMap(scope map[string]interface{}) (map[string]interface{}, error) {
	type domainReq struct {
		ID   *string `json:"id,omitempty"`
		Name *string `json:"name,omitempty"`
	}

	type projectReq struct {
		Domain *domainReq `json:"domain,omitempty"`
		Name   *string    `json:"name,omitempty"`
		ID     *string    `json:"id,omitempty"`
	}

	type userReq struct {
		ID       *string    `json:"id,omitempty"`
		Name     *string    `json:"name,omitempty"`
		Password string     `json:"password"`
		Domain   *domainReq `json:"domain,omitempty"`
	}

	type passwordReq struct {
		User userReq `json:"user"`
	}

	type tokenReq struct {
		ID string `json:"id"`
	}

	type identityReq struct {
		Methods  []string     `json:"methods"`
		Password *passwordReq `json:"password,omitempty"`
		Token    *tokenReq    `json:"token,omitempty"`
	}

	type authReq struct {
		Identity identityReq `json:"identity"`
	}

	type request struct {
		Auth authReq `json:"auth"`
	}

	// Populate the request structure based on the provided arguments. Create and return an error
	// if insufficient or incompatible information is present.
	var req request

	if opts.Password == "" {
		if opts.TokenID != "" {
			// Because we aren't using password authentication, it's an error to also provide any of the user-based authentication
			// parameters.
			if opts.Username != "" {
				return nil, ErrUsernameWithToken{}
			}
			if opts.UserID != "" {
				return nil, ErrUserIDWithToken{}
			}
			if opts.DomainID != "" {
				return nil, ErrDomainIDWithToken{}
			}
			if opts.DomainName != "" {
				return nil, ErrDomainNameWithToken{}
			}

			// Configure the request for Token authentication.
			req.Auth.Identity.Methods = []string{"token"}
			req.Auth.Identity.Token = &tokenReq{
				ID: opts.TokenID,
			}
		} else {
			// If no password or token ID are available, authentication can't continue.
			return nil, ErrMissingPassword{}
		}
	} else {
		// Password authentication.
		req.Auth.Identity.Methods = []string{"password"}

		// At least one of Username and UserID must be specified.
		if opts.Username == "" && opts.UserID == "" {
			return nil, ErrUsernameOrUserID{}
		}

		if opts.Username != "" {
			// If Username is provided, UserID may not be provided.
			if opts.UserID != "" {
				return nil, ErrUsernameOrUserID{}
			}

			// Either DomainID or DomainName must also be specified.
			if opts.DomainID == "" && opts.DomainName == "" {
				return nil, ErrDomainIDOrDomainName{}
			}

			if opts.DomainID != "" {
				if opts.DomainName != "" {
					return nil, ErrDomainIDOrDomainName{}
				}

				// Configure the request for Username and Password authentication with a DomainID.
				req.Auth.Identity.Password = &passwordReq{
					User: userReq{
						Name:     &opts.Username,
						Password: opts.Password,
						Domain:   &domainReq{ID: &opts.DomainID},
					},
				}
			}

			if opts.DomainName != "" {
				// Configure the request for Username and Password authentication with a DomainName.
				req.Auth.Identity.Password = &passwordReq{
					User: userReq{
						Name:     &opts.Username,
						Password: opts.Password,
						Domain:   &domainReq{Name: &opts.DomainName},
					},
				}
			}
		}

		if opts.UserID != "" {
			// If UserID is specified, neither DomainID nor DomainName may be.
			if opts.DomainID != "" {
				return nil, ErrDomainIDWithUserID{}
			}
			if opts.DomainName != "" {
				return nil, ErrDomainNameWithUserID{}
			}

			// Configure the request for UserID and Password authentication.
			req.Auth.Identity.Password = &passwordReq{
				User: userReq{ID: &opts.UserID, Password: opts.Password},
			}
		}
	}

	b, err := BuildRequestBody(req, "")
	if err != nil {
		return nil, err
	}

	if len(scope) != 0 {
		b["auth"].(map[string]interface{})["scope"] = scope
	}

	return b, nil
}

func (opts *AuthOptions) ToTokenV3ScopeMap() (map[string]interface{}, error) {

	var scope struct {
		ProjectID   string
		ProjectName string
		DomainID    string
		DomainName  string
	}

	if opts.TenantID != "" {
		scope.ProjectID = opts.TenantID
	} else {
		if opts.TenantName != "" {
			scope.ProjectName = opts.TenantName
			scope.DomainID = opts.DomainID
			scope.DomainName = opts.DomainName
		}
	}

	if scope.ProjectName != "" {
		// ProjectName provided: either DomainID or DomainName must also be supplied.
		// ProjectID may not be supplied.
		if scope.DomainID == "" && scope.DomainName == "" {
			return nil, ErrScopeDomainIDOrDomainName{}
		}
		if scope.ProjectID != "" {
			return nil, ErrScopeProjectIDOrProjectName{}
		}

		if scope.DomainID != "" {
			// ProjectName + DomainID
			return map[string]interface{}{
				"project": map[string]interface{}{
					"name":   &scope.ProjectName,
					"domain": map[string]interface{}{"id": &scope.DomainID},
				},
			}, nil
		}

		if scope.DomainName != "" {
			// ProjectName + DomainName
			return map[string]interface{}{
				"project": map[string]interface{}{
					"name":   &scope.ProjectName,
					"domain": map[string]interface{}{"name": &scope.DomainName},
				},
			}, nil
		}
	} else if scope.ProjectID != "" {
		// ProjectID provided. ProjectName, DomainID, and DomainName may not be provided.
		if scope.DomainID != "" {
			return nil, ErrScopeProjectIDAlone{}
		}
		if scope.DomainName != "" {
			return nil, ErrScopeProjectIDAlone{}
		}

		// ProjectID
		return map[string]interface{}{
			"project": map[string]interface{}{
				"id": &scope.ProjectID,
			},
		}, nil
	} else if scope.DomainID != "" {
		// DomainID provided. ProjectID, ProjectName, and DomainName may not be provided.
		if scope.DomainName != "" {
			return nil, ErrScopeDomainIDOrDomainName{}
		}

		// DomainID
		return map[string]interface{}{
			"domain": map[string]interface{}{
				"id": &scope.DomainID,
			},
		}, nil
	} else if scope.DomainName != "" {
		return nil, ErrScopeDomainName{}
	}

	return nil, nil
}

func (opts AuthOptions) CanReauth() bool {
	return opts.AllowReauth
}
