package testing

import (
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/identity/v3/extensions/trusts"
	"github.com/gophercloud/gophercloud/openstack/identity/v3/tokens"
	"github.com/gophercloud/gophercloud/testhelper"
)

// HandleCreateTokenWithTrustID verifies that providing certain AuthOptions and Scope results in an expected JSON structure.
func HandleCreateTokenWithTrustID(t *testing.T, options tokens.AuthOptionsBuilder, requestJSON string) {
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
        "expires_at": "2013-02-27T18:30:59.999999Z",
        "issued_at": "2013-02-27T16:30:59.999999Z",
        "methods": [
            "password"
        ],
        "OS-TRUST:trust": {
            "id": "fe0aef",
            "impersonation": false,
						"redelegated_trust_id": "3ba234",
						"redelegation_count": 2,
            "links": {
                "self": "http://example.com/identity/v3/trusts/fe0aef"
            },
            "trustee_user": {
                "id": "0ca8f6",
                "links": {
                    "self": "http://example.com/identity/v3/users/0ca8f6"
                }
            },
            "trustor_user": {
                "id": "bd263c",
                "links": {
                    "self": "http://example.com/identity/v3/users/bd263c"
                }
            }
        },
        "user": {
            "domain": {
                "id": "1789d1",
                "links": {
                    "self": "http://example.com/identity/v3/domains/1789d1"
                },
                "name": "example.com"
            },
            "email": "joe@example.com",
            "id": "0ca8f6",
            "links": {
                "self": "http://example.com/identity/v3/users/0ca8f6"
            },
            "name": "Joe"
        }
    }
}`)
	})

	var actual trusts.TokenExt
	err := tokens.Create(&client, options).ExtractInto(&actual)
	if err != nil {
		t.Errorf("Create returned an error: %v", err)
	}
	expected := trusts.TokenExt{
		Token: trusts.Token{
			Token: tokens.Token{
				ExpiresAt: gophercloud.JSONRFC3339Milli(time.Date(2013, 02, 27, 18, 30, 59, 999999000, time.UTC)),
			},
			Trust: trusts.Trust{
				ID:            "fe0aef",
				Impersonation: false,
				TrusteeUser: trusts.TrusteeUser{
					ID: "0ca8f6",
				},
				TrustorUser: trusts.TrustorUser{
					ID: "bd263c",
				},
				RedelegatedTrustID: "3ba234",
				RedelegationCount:  2,
			},
		},
	}
	testhelper.AssertDeepEquals(t, expected, actual)
}
