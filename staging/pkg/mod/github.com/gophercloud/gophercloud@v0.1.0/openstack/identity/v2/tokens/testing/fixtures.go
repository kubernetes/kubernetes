package testing

import (
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/gophercloud/gophercloud/openstack/identity/v2/tenants"
	"github.com/gophercloud/gophercloud/openstack/identity/v2/tokens"
	th "github.com/gophercloud/gophercloud/testhelper"
	thclient "github.com/gophercloud/gophercloud/testhelper/client"
)

// ExpectedToken is the token that should be parsed from TokenCreationResponse.
var ExpectedToken = &tokens.Token{
	ID:        "aaaabbbbccccdddd",
	ExpiresAt: time.Date(2014, time.January, 31, 15, 30, 58, 0, time.UTC),
	Tenant: tenants.Tenant{
		ID:          "fc394f2ab2df4114bde39905f800dc57",
		Name:        "test",
		Description: "There are many tenants. This one is yours.",
		Enabled:     true,
	},
}

// ExpectedServiceCatalog is the service catalog that should be parsed from TokenCreationResponse.
var ExpectedServiceCatalog = &tokens.ServiceCatalog{
	Entries: []tokens.CatalogEntry{
		{
			Name: "inscrutablewalrus",
			Type: "something",
			Endpoints: []tokens.Endpoint{
				{
					PublicURL: "http://something0:1234/v2/",
					Region:    "region0",
				},
				{
					PublicURL: "http://something1:1234/v2/",
					Region:    "region1",
				},
			},
		},
		{
			Name: "arbitrarypenguin",
			Type: "else",
			Endpoints: []tokens.Endpoint{
				{
					PublicURL: "http://else0:4321/v3/",
					Region:    "region0",
				},
			},
		},
	},
}

// ExpectedUser is the token that should be parsed from TokenGetResponse.
var ExpectedUser = &tokens.User{
	ID:       "a530fefc3d594c4ba2693a4ecd6be74e",
	Name:     "apiserver",
	Roles:    []tokens.Role{tokens.Role{Name: "member"}, tokens.Role{Name: "service"}},
	UserName: "apiserver",
}

// TokenCreationResponse is a JSON response that contains ExpectedToken and ExpectedServiceCatalog.
const TokenCreationResponse = `
{
	"access": {
		"token": {
			"issued_at": "2014-01-30T15:30:58.000000Z",
			"expires": "2014-01-31T15:30:58Z",
			"id": "aaaabbbbccccdddd",
			"tenant": {
				"description": "There are many tenants. This one is yours.",
				"enabled": true,
				"id": "fc394f2ab2df4114bde39905f800dc57",
				"name": "test"
			}
		},
		"serviceCatalog": [
			{
				"endpoints": [
					{
						"publicURL": "http://something0:1234/v2/",
						"region": "region0"
					},
					{
						"publicURL": "http://something1:1234/v2/",
						"region": "region1"
					}
				],
				"type": "something",
				"name": "inscrutablewalrus"
			},
			{
				"endpoints": [
					{
						"publicURL": "http://else0:4321/v3/",
						"region": "region0"
					}
				],
				"type": "else",
				"name": "arbitrarypenguin"
			}
		]
	}
}
`

// TokenGetResponse is a JSON response that contains ExpectedToken and ExpectedUser.
const TokenGetResponse = `
{
    "access": {
		"token": {
			"issued_at": "2014-01-30T15:30:58.000000Z",
			"expires": "2014-01-31T15:30:58Z",
			"id": "aaaabbbbccccdddd",
			"tenant": {
				"description": "There are many tenants. This one is yours.",
				"enabled": true,
				"id": "fc394f2ab2df4114bde39905f800dc57",
				"name": "test"
			}
		},
        "serviceCatalog": [],
		"user": {
            "id": "a530fefc3d594c4ba2693a4ecd6be74e",
            "name": "apiserver",
            "roles": [
                {
                    "name": "member"
                },
                {
                    "name": "service"
                }
            ],
            "roles_links": [],
            "username": "apiserver"
        }
    }
}`

// HandleTokenPost expects a POST against a /tokens handler, ensures that the request body has been
// constructed properly given certain auth options, and returns the result.
func HandleTokenPost(t *testing.T, requestJSON string) {
	th.Mux.HandleFunc("/tokens", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		if requestJSON != "" {
			th.TestJSONRequest(t, r, requestJSON)
		}

		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, TokenCreationResponse)
	})
}

// HandleTokenGet expects a Get against a /tokens handler, ensures that the request body has been
// constructed properly given certain auth options, and returns the result.
func HandleTokenGet(t *testing.T, token string) {
	th.Mux.HandleFunc("/tokens/"+token, func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestHeader(t, r, "X-Auth-Token", thclient.TokenID)

		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, TokenGetResponse)
	})
}

// IsSuccessful ensures that a CreateResult was successful and contains the correct token and
// service catalog.
func IsSuccessful(t *testing.T, result tokens.CreateResult) {
	token, err := result.ExtractToken()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, ExpectedToken, token)

	serviceCatalog, err := result.ExtractServiceCatalog()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, ExpectedServiceCatalog, serviceCatalog)
}

// GetIsSuccessful ensures that a GetResult was successful and contains the correct token and
// User Info.
func GetIsSuccessful(t *testing.T, result tokens.GetResult) {
	token, err := result.ExtractToken()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, ExpectedToken, token)

	user, err := result.ExtractUser()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, ExpectedUser, user)
}
