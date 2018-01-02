package testing

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/gophercloud/gophercloud/openstack/identity/v3/tokens"
	"github.com/gophercloud/gophercloud/testhelper"
)

// HandleCreateTokenWithTrustID verifies that providing certain AuthOptions and Scope results in an expected JSON structure.
func HandleCreateTokenWithTrustID(t *testing.T, options tokens.AuthOptionsBuilder, requestJSON string) {
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
}
