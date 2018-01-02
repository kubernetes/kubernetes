package testing

import (
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/identity/v3/tokens"
	"github.com/gophercloud/gophercloud/testhelper"
)

// authTokenPost verifies that providing certain AuthOptions and Scope results in an expected JSON structure.
func authTokenPost(t *testing.T, options tokens.AuthOptions, scope *tokens.Scope, requestJSON string) {
	testhelper.SetupHTTP()
	defer testhelper.TeardownHTTP()

	client := gophercloud.ServiceClient{
		ProviderClient: &gophercloud.ProviderClient{},
		Endpoint:       testhelper.Endpoint(),
	}

	testhelper.Mux.HandleFunc("/auth/tokens", func(w http.ResponseWriter, r *http.Request) {
		testhelper.TestMethod(t, r, "POST")
		testhelper.TestHeader(t, r, "Content-Type", "application/json")
		testhelper.TestHeader(t, r, "Accept", "application/json")
		testhelper.TestJSONRequest(t, r, requestJSON)

		w.WriteHeader(http.StatusCreated)
		fmt.Fprintf(w, `{
			"token": {
				"expires_at": "2014-10-02T13:45:00.000000Z"
			}
		}`)
	})

	if scope != nil {
		options.Scope = *scope
	}

	expected := &tokens.Token{
		ExpiresAt: time.Date(2014, 10, 2, 13, 45, 0, 0, time.UTC),
	}
	actual, err := tokens.Create(&client, &options).Extract()
	testhelper.AssertNoErr(t, err)
	testhelper.CheckDeepEquals(t, expected, actual)
}

func authTokenPostErr(t *testing.T, options tokens.AuthOptions, scope *tokens.Scope, includeToken bool, expectedErr error) {
	testhelper.SetupHTTP()
	defer testhelper.TeardownHTTP()

	client := gophercloud.ServiceClient{
		ProviderClient: &gophercloud.ProviderClient{},
		Endpoint:       testhelper.Endpoint(),
	}
	if includeToken {
		client.TokenID = "abcdef123456"
	}

	if scope != nil {
		options.Scope = *scope
	}

	_, err := tokens.Create(&client, &options).Extract()
	if err == nil {
		t.Errorf("Create did NOT return an error")
	}
	if err != expectedErr {
		t.Errorf("Create returned an unexpected error: wanted %v, got %v", expectedErr, err)
	}
}

func TestCreateUserIDAndPassword(t *testing.T) {
	authTokenPost(t, tokens.AuthOptions{UserID: "me", Password: "squirrel!"}, nil, `
		{
			"auth": {
				"identity": {
					"methods": ["password"],
					"password": {
						"user": { "id": "me", "password": "squirrel!" }
					}
				}
			}
		}
	`)
}

func TestCreateUsernameDomainIDPassword(t *testing.T) {
	authTokenPost(t, tokens.AuthOptions{Username: "fakey", Password: "notpassword", DomainID: "abc123"}, nil, `
		{
			"auth": {
				"identity": {
					"methods": ["password"],
					"password": {
						"user": {
							"domain": {
								"id": "abc123"
							},
							"name": "fakey",
							"password": "notpassword"
						}
					}
				}
			}
		}
	`)
}

func TestCreateUsernameDomainNamePassword(t *testing.T) {
	authTokenPost(t, tokens.AuthOptions{Username: "frank", Password: "swordfish", DomainName: "spork.net"}, nil, `
		{
			"auth": {
				"identity": {
					"methods": ["password"],
					"password": {
						"user": {
							"domain": {
								"name": "spork.net"
							},
							"name": "frank",
							"password": "swordfish"
						}
					}
				}
			}
		}
	`)
}

func TestCreateTokenID(t *testing.T) {
	authTokenPost(t, tokens.AuthOptions{TokenID: "12345abcdef"}, nil, `
		{
			"auth": {
				"identity": {
					"methods": ["token"],
					"token": {
						"id": "12345abcdef"
					}
				}
			}
		}
	`)
}

func TestCreateProjectIDScope(t *testing.T) {
	options := tokens.AuthOptions{UserID: "fenris", Password: "g0t0h311"}
	scope := &tokens.Scope{ProjectID: "123456"}
	authTokenPost(t, options, scope, `
		{
			"auth": {
				"identity": {
					"methods": ["password"],
					"password": {
						"user": {
							"id": "fenris",
							"password": "g0t0h311"
						}
					}
				},
				"scope": {
					"project": {
						"id": "123456"
					}
				}
			}
		}
	`)
}

func TestCreateDomainIDScope(t *testing.T) {
	options := tokens.AuthOptions{UserID: "fenris", Password: "g0t0h311"}
	scope := &tokens.Scope{DomainID: "1000"}
	authTokenPost(t, options, scope, `
		{
			"auth": {
				"identity": {
					"methods": ["password"],
					"password": {
						"user": {
							"id": "fenris",
							"password": "g0t0h311"
						}
					}
				},
				"scope": {
					"domain": {
						"id": "1000"
					}
				}
			}
		}
	`)
}

func TestCreateDomainNameScope(t *testing.T) {
	options := tokens.AuthOptions{UserID: "fenris", Password: "g0t0h311"}
	scope := &tokens.Scope{DomainName: "evil-plans"}
	authTokenPost(t, options, scope, `
                {
                        "auth": {
                                "identity": {
                                        "methods": ["password"],
                                        "password": {
                                                "user": {
                                                        "id": "fenris",
                                                        "password": "g0t0h311"
                                                }
                                        }
                                },
                                "scope": {
                                        "domain": {
                                                "name": "evil-plans"
                                        }
                                }
                        }
                }
        `)
}

func TestCreateProjectNameAndDomainIDScope(t *testing.T) {
	options := tokens.AuthOptions{UserID: "fenris", Password: "g0t0h311"}
	scope := &tokens.Scope{ProjectName: "world-domination", DomainID: "1000"}
	authTokenPost(t, options, scope, `
		{
			"auth": {
				"identity": {
					"methods": ["password"],
					"password": {
						"user": {
							"id": "fenris",
							"password": "g0t0h311"
						}
					}
				},
				"scope": {
					"project": {
						"domain": {
							"id": "1000"
						},
						"name": "world-domination"
					}
				}
			}
		}
	`)
}

func TestCreateProjectNameAndDomainNameScope(t *testing.T) {
	options := tokens.AuthOptions{UserID: "fenris", Password: "g0t0h311"}
	scope := &tokens.Scope{ProjectName: "world-domination", DomainName: "evil-plans"}
	authTokenPost(t, options, scope, `
		{
			"auth": {
				"identity": {
					"methods": ["password"],
					"password": {
						"user": {
							"id": "fenris",
							"password": "g0t0h311"
						}
					}
				},
				"scope": {
					"project": {
						"domain": {
							"name": "evil-plans"
						},
						"name": "world-domination"
					}
				}
			}
		}
	`)
}

func TestCreateExtractsTokenFromResponse(t *testing.T) {
	testhelper.SetupHTTP()
	defer testhelper.TeardownHTTP()

	client := gophercloud.ServiceClient{
		ProviderClient: &gophercloud.ProviderClient{},
		Endpoint:       testhelper.Endpoint(),
	}

	testhelper.Mux.HandleFunc("/auth/tokens", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Add("X-Subject-Token", "aaa111")

		w.WriteHeader(http.StatusCreated)
		fmt.Fprintf(w, `{
			"token": {
				"expires_at": "2014-10-02T13:45:00.000000Z"
			}
		}`)
	})

	options := tokens.AuthOptions{UserID: "me", Password: "shhh"}
	token, err := tokens.Create(&client, &options).Extract()
	if err != nil {
		t.Fatalf("Create returned an error: %v", err)
	}

	if token.ID != "aaa111" {
		t.Errorf("Expected token to be aaa111, but was %s", token.ID)
	}
}

func TestCreateFailureEmptyAuth(t *testing.T) {
	authTokenPostErr(t, tokens.AuthOptions{}, nil, false, gophercloud.ErrMissingPassword{})
}

func TestCreateFailureTokenIDUsername(t *testing.T) {
	authTokenPostErr(t, tokens.AuthOptions{Username: "something", TokenID: "12345"}, nil, true, gophercloud.ErrUsernameWithToken{})
}

func TestCreateFailureTokenIDUserID(t *testing.T) {
	authTokenPostErr(t, tokens.AuthOptions{UserID: "something", TokenID: "12345"}, nil, true, gophercloud.ErrUserIDWithToken{})
}

func TestCreateFailureTokenIDDomainID(t *testing.T) {
	authTokenPostErr(t, tokens.AuthOptions{DomainID: "something", TokenID: "12345"}, nil, true, gophercloud.ErrDomainIDWithToken{})
}

func TestCreateFailureTokenIDDomainName(t *testing.T) {
	authTokenPostErr(t, tokens.AuthOptions{DomainName: "something", TokenID: "12345"}, nil, true, gophercloud.ErrDomainNameWithToken{})
}

func TestCreateFailureMissingUser(t *testing.T) {
	options := tokens.AuthOptions{Password: "supersecure"}
	authTokenPostErr(t, options, nil, false, gophercloud.ErrUsernameOrUserID{})
}

func TestCreateFailureBothUser(t *testing.T) {
	options := tokens.AuthOptions{
		Password: "supersecure",
		Username: "oops",
		UserID:   "redundancy",
	}
	authTokenPostErr(t, options, nil, false, gophercloud.ErrUsernameOrUserID{})
}

func TestCreateFailureMissingDomain(t *testing.T) {
	options := tokens.AuthOptions{
		Password: "supersecure",
		Username: "notuniqueenough",
	}
	authTokenPostErr(t, options, nil, false, gophercloud.ErrDomainIDOrDomainName{})
}

func TestCreateFailureBothDomain(t *testing.T) {
	options := tokens.AuthOptions{
		Password:   "supersecure",
		Username:   "someone",
		DomainID:   "hurf",
		DomainName: "durf",
	}
	authTokenPostErr(t, options, nil, false, gophercloud.ErrDomainIDOrDomainName{})
}

func TestCreateFailureUserIDDomainID(t *testing.T) {
	options := tokens.AuthOptions{
		UserID:   "100",
		Password: "stuff",
		DomainID: "oops",
	}
	authTokenPostErr(t, options, nil, false, gophercloud.ErrDomainIDWithUserID{})
}

func TestCreateFailureUserIDDomainName(t *testing.T) {
	options := tokens.AuthOptions{
		UserID:     "100",
		Password:   "sssh",
		DomainName: "oops",
	}
	authTokenPostErr(t, options, nil, false, gophercloud.ErrDomainNameWithUserID{})
}

func TestCreateFailureScopeProjectNameAlone(t *testing.T) {
	options := tokens.AuthOptions{UserID: "myself", Password: "swordfish"}
	scope := &tokens.Scope{ProjectName: "notenough"}
	authTokenPostErr(t, options, scope, false, gophercloud.ErrScopeDomainIDOrDomainName{})
}

func TestCreateFailureScopeProjectNameAndID(t *testing.T) {
	options := tokens.AuthOptions{UserID: "myself", Password: "swordfish"}
	scope := &tokens.Scope{ProjectName: "whoops", ProjectID: "toomuch", DomainID: "1234"}
	authTokenPostErr(t, options, scope, false, gophercloud.ErrScopeProjectIDOrProjectName{})
}

func TestCreateFailureScopeProjectIDAndDomainID(t *testing.T) {
	options := tokens.AuthOptions{UserID: "myself", Password: "swordfish"}
	scope := &tokens.Scope{ProjectID: "toomuch", DomainID: "notneeded"}
	authTokenPostErr(t, options, scope, false, gophercloud.ErrScopeProjectIDAlone{})
}

func TestCreateFailureScopeProjectIDAndDomainNAme(t *testing.T) {
	options := tokens.AuthOptions{UserID: "myself", Password: "swordfish"}
	scope := &tokens.Scope{ProjectID: "toomuch", DomainName: "notneeded"}
	authTokenPostErr(t, options, scope, false, gophercloud.ErrScopeProjectIDAlone{})
}

func TestCreateFailureScopeDomainIDAndDomainName(t *testing.T) {
	options := tokens.AuthOptions{UserID: "myself", Password: "swordfish"}
	scope := &tokens.Scope{DomainID: "toomuch", DomainName: "notneeded"}
	authTokenPostErr(t, options, scope, false, gophercloud.ErrScopeDomainIDOrDomainName{})
}

/*
func TestCreateFailureEmptyScope(t *testing.T) {
	options := tokens.AuthOptions{UserID: "myself", Password: "swordfish"}
	scope := &tokens.Scope{}
	authTokenPostErr(t, options, scope, false, gophercloud.ErrScopeEmpty{})
}
*/

func TestGetRequest(t *testing.T) {
	testhelper.SetupHTTP()
	defer testhelper.TeardownHTTP()

	client := gophercloud.ServiceClient{
		ProviderClient: &gophercloud.ProviderClient{
			TokenID: "12345abcdef",
		},
		Endpoint: testhelper.Endpoint(),
	}

	testhelper.Mux.HandleFunc("/auth/tokens", func(w http.ResponseWriter, r *http.Request) {
		testhelper.TestMethod(t, r, "GET")
		testhelper.TestHeader(t, r, "Content-Type", "")
		testhelper.TestHeader(t, r, "Accept", "application/json")
		testhelper.TestHeader(t, r, "X-Auth-Token", "12345abcdef")
		testhelper.TestHeader(t, r, "X-Subject-Token", "abcdef12345")

		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `
			{ "token": { "expires_at": "2014-08-29T13:10:01.000000Z" } }
		`)
	})

	token, err := tokens.Get(&client, "abcdef12345").Extract()
	if err != nil {
		t.Errorf("Info returned an error: %v", err)
	}

	expected, _ := time.Parse(time.UnixDate, "Fri Aug 29 13:10:01 UTC 2014")
	if token.ExpiresAt != expected {
		t.Errorf("Expected expiration time %s, but was %s", expected.Format(time.UnixDate), time.Time(token.ExpiresAt).Format(time.UnixDate))
	}
}

func prepareAuthTokenHandler(t *testing.T, expectedMethod string, status int) gophercloud.ServiceClient {
	client := gophercloud.ServiceClient{
		ProviderClient: &gophercloud.ProviderClient{
			TokenID: "12345abcdef",
		},
		Endpoint: testhelper.Endpoint(),
	}

	testhelper.Mux.HandleFunc("/auth/tokens", func(w http.ResponseWriter, r *http.Request) {
		testhelper.TestMethod(t, r, expectedMethod)
		testhelper.TestHeader(t, r, "Content-Type", "")
		testhelper.TestHeader(t, r, "Accept", "application/json")
		testhelper.TestHeader(t, r, "X-Auth-Token", "12345abcdef")
		testhelper.TestHeader(t, r, "X-Subject-Token", "abcdef12345")

		w.WriteHeader(status)
	})

	return client
}

func TestValidateRequestSuccessful(t *testing.T) {
	testhelper.SetupHTTP()
	defer testhelper.TeardownHTTP()
	client := prepareAuthTokenHandler(t, "HEAD", http.StatusNoContent)

	ok, err := tokens.Validate(&client, "abcdef12345")
	if err != nil {
		t.Errorf("Unexpected error from Validate: %v", err)
	}

	if !ok {
		t.Errorf("Validate returned false for a valid token")
	}
}

func TestValidateRequestFailure(t *testing.T) {
	testhelper.SetupHTTP()
	defer testhelper.TeardownHTTP()
	client := prepareAuthTokenHandler(t, "HEAD", http.StatusNotFound)

	ok, err := tokens.Validate(&client, "abcdef12345")
	if err != nil {
		t.Errorf("Unexpected error from Validate: %v", err)
	}

	if ok {
		t.Errorf("Validate returned true for an invalid token")
	}
}

func TestValidateRequestError(t *testing.T) {
	testhelper.SetupHTTP()
	defer testhelper.TeardownHTTP()
	client := prepareAuthTokenHandler(t, "HEAD", http.StatusMethodNotAllowed)

	_, err := tokens.Validate(&client, "abcdef12345")
	if err == nil {
		t.Errorf("Missing expected error from Validate")
	}
}

func TestRevokeRequestSuccessful(t *testing.T) {
	testhelper.SetupHTTP()
	defer testhelper.TeardownHTTP()
	client := prepareAuthTokenHandler(t, "DELETE", http.StatusNoContent)

	res := tokens.Revoke(&client, "abcdef12345")
	testhelper.AssertNoErr(t, res.Err)
}

func TestRevokeRequestError(t *testing.T) {
	testhelper.SetupHTTP()
	defer testhelper.TeardownHTTP()
	client := prepareAuthTokenHandler(t, "DELETE", http.StatusNotFound)

	res := tokens.Revoke(&client, "abcdef12345")
	if res.Err == nil {
		t.Errorf("Missing expected error from Revoke")
	}
}

func TestNoTokenInResponse(t *testing.T) {
	testhelper.SetupHTTP()
	defer testhelper.TeardownHTTP()

	client := gophercloud.ServiceClient{
		ProviderClient: &gophercloud.ProviderClient{},
		Endpoint:       testhelper.Endpoint(),
	}

	testhelper.Mux.HandleFunc("/auth/tokens", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusCreated)
		fmt.Fprintf(w, `{}`)
	})

	options := tokens.AuthOptions{UserID: "me", Password: "squirrel!"}
	_, err := tokens.Create(&client, &options).Extract()
	testhelper.AssertNoErr(t, err)
}
