package tokens

import (
	"net/http"

	"github.com/rackspace/gophercloud"
)

// Scope allows a created token to be limited to a specific domain or project.
type Scope struct {
	ProjectID   string
	ProjectName string
	DomainID    string
	DomainName  string
}

func subjectTokenHeaders(c *gophercloud.ServiceClient, subjectToken string) map[string]string {
	return map[string]string{
		"X-Subject-Token": subjectToken,
	}
}

// AuthOptionsV3er describes any argument that may be passed to the Create call.
type AuthOptionsV3er interface {

        // ToTokenCreateMap assembles the Create request body, returning an error if parameters are
        // missing or inconsistent.
        ToAuthOptionsV3Map(c *gophercloud.ServiceClient, scope *Scope) (map[string]interface{}, error)
}

// AuthOptions wraps a gophercloud AuthOptions in order to adhere to the AuthOptionsV3er
// interface.
type AuthOptions struct {
        gophercloud.AuthOptions
}

func (options AuthOptions) ToAuthOptionsV3Map(c *gophercloud.ServiceClient, scope *Scope) (map[string]interface{}, error) {
	// tokens3.Create logic

	// Populate the request structure based on the provided arguments. Create and return an error
	// if insufficient or incompatible information is present.
	authMap := make(map[string]interface{})

	// Test first for unrecognized arguments.
	if options.APIKey != "" {
		return nil, ErrAPIKeyProvided
	}
	if options.TenantID != "" {
		return nil, ErrTenantIDProvided
	}
	if options.TenantName != "" {
		return nil, ErrTenantNameProvided
	}

	if options.Password == "" {
		if options.TokenID != "" {
			c.TokenID = options.TokenID
		}
		if c.TokenID != "" {
			// Because we aren't using password authentication, it's an error to also provide any of the user-based authentication
			// parameters.
			if options.Username != "" {
				return nil, ErrUsernameWithToken
			}
			if options.UserID != "" {
				return nil, ErrUserIDWithToken
			}

			// Configure the request for Token authentication.
                        authMap["identity"] = map[string]interface{}{
                                "methods": []string{"token"},
                                "token": map[string]interface{}{
                                        "id": c.TokenID,
				},
			}

		} else {
			// If no password or token ID are available, authentication can't continue.
			return nil, ErrMissingPassword
		}
	} else {
		// Password authentication.

		// At least one of Username and UserID must be specified.
		if options.Username == "" && options.UserID == "" {
			return nil, ErrUsernameOrUserID
		}

		if options.Username != "" {
			// If Username is provided, UserID may not be provided.
			if options.UserID != "" {
				return nil, ErrUsernameOrUserID
			}

			// Either DomainID or DomainName must also be specified.
			if options.DomainID == "" && options.DomainName == "" {
				return nil, ErrDomainIDOrDomainName
			}

			if options.DomainID != "" {
				if options.DomainName != "" {
					return nil, ErrDomainIDOrDomainName
				}

				// Configure the request for Username and Password authentication with a DomainID.
                                authMap["identity"] = map[string]interface{}{
					"methods": []string{"password"},
                                	"password" : map[string]interface{}{
                                                "user": map[string]interface{}{
                                                        "name": &options.Username,
                                                        "password": options.Password,
                                                        "domain": map[string]interface{}{
                                                                "id": &options.DomainID,
                                                        },
                                                },
					},
				}

			}

			if options.DomainName != "" {
				// Configure the request for Username and Password authentication with a DomainName.
                                authMap["identity"] = map[string]interface{}{
					"methods": []string{"password"},
                                        "password": map[string]interface{}{
                                                 "user": map[string]interface{}{
                                                         "name": &options.Username,
                                                         "password": options.Password,
                                                         "domain": map[string]interface{}{
                                                                 "name": &options.DomainName,
                                                         },
                                                },
                                         },
                                 }

			}
		}

		if options.UserID != "" {
			// If UserID is specified, neither DomainID nor DomainName may be.
			if options.DomainID != "" {
				return nil, ErrDomainIDWithUserID
			}
			if options.DomainName != "" {
				return nil, ErrDomainNameWithUserID
			}

			// Configure the request for UserID and Password authentication.
                        authMap["identity"] = map[string]interface{}{
				"methods": []string{"password"},
                                "password" : map[string]interface{}{
                                        "user": map[string]interface{}{
                                                "id": &options.UserID,
                                                "password": options.Password,
                                        },
                                },
			}

		}
	}

	// Add a "scope" element if a Scope has been provided.
	if scope != nil {
		if scope.ProjectName != "" {
			// ProjectName provided: either DomainID or DomainName must also be supplied.
			// ProjectID may not be supplied.
			if scope.DomainID == "" && scope.DomainName == "" {
				return nil, ErrScopeDomainIDOrDomainName
			}
			if scope.ProjectID != "" {
				return nil, ErrScopeProjectIDOrProjectName
			}

			if scope.DomainID != "" {
				// ProjectName + DomainID
                                authMap["scope"] = map[string]interface{}{
                                        "project": map[string]interface{}{
                                                "domain": map[string]interface{}{
                                                        "id": &scope.DomainID,
                                                        },
                                                "name": &scope.ProjectName,
                                        },
				}
			}

			if scope.DomainName != "" {
				// ProjectName + DomainName
                                authMap["scope"] = map[string]interface{}{
                                        "project": map[string]interface{}{
                                                "domain": map[string]interface{}{
                                                        "name": &scope.DomainName,
                                                        },
                                                "name": &scope.ProjectName,
                                        },
				}
			}
		} else if scope.ProjectID != "" {
			// ProjectID provided. ProjectName, DomainID, and DomainName may not be provided.
			if scope.DomainID != "" {
				return nil, ErrScopeProjectIDAlone
			}
			if scope.DomainName != "" {
				return nil, ErrScopeProjectIDAlone
			}

			// ProjectID
                        authMap["scope"] = map[string]interface{}{
                                "project": map[string]interface{}{
                                        "id": &scope.ProjectID,
                                        },
			}
		} else if scope.DomainID != "" {
			// DomainID provided. ProjectID, ProjectName, and DomainName may not be provided.
			if scope.DomainName != "" {
				return nil, ErrScopeDomainIDOrDomainName
			}

			// DomainID
                        authMap["scope"] = map[string]interface{}{
                                 "domain": map[string]interface{}{
                                         "id": &scope.DomainID,
                                         },
			}
		} else if scope.DomainName != "" {
			return nil, ErrScopeDomainName
		} else {
			return nil, ErrScopeEmpty
		}
	}
	return map[string]interface{}{"auth": authMap}, nil
}

// Create authenticates and either generates a new token, or changes the Scope of an existing token.
func Create(c *gophercloud.ServiceClient, options AuthOptionsV3er, scope *Scope) CreateResult {
        request, err := options.ToAuthOptionsV3Map(c, scope)
        if err != nil {
                return CreateResult{commonResult{gophercloud.Result{Err: err}}}
        }

	var result CreateResult
	var response *http.Response
	response, result.Err = c.Post(tokenURL(c), request, &result.Body, nil)
	if result.Err != nil {
		return result
	}
	result.Header = response.Header
	return result
}

// Get validates and retrieves information about another token.
func Get(c *gophercloud.ServiceClient, token string) GetResult {
	var result GetResult
	var response *http.Response
	response, result.Err = c.Get(tokenURL(c), &result.Body, &gophercloud.RequestOpts{
		MoreHeaders: subjectTokenHeaders(c, token),
		OkCodes:     []int{200, 203},
	})
	if result.Err != nil {
		return result
	}
	result.Header = response.Header
	return result
}

// Validate determines if a specified token is valid or not.
func Validate(c *gophercloud.ServiceClient, token string) (bool, error) {
	response, err := c.Request("HEAD", tokenURL(c), gophercloud.RequestOpts{
		MoreHeaders: subjectTokenHeaders(c, token),
		OkCodes:     []int{204, 404},
	})
	if err != nil {
		return false, err
	}

	return response.StatusCode == 204, nil
}

// Revoke immediately makes specified token invalid.
func Revoke(c *gophercloud.ServiceClient, token string) RevokeResult {
	var res RevokeResult
	_, res.Err = c.Delete(tokenURL(c), &gophercloud.RequestOpts{
		MoreHeaders: subjectTokenHeaders(c, token),
	})
	return res
}
