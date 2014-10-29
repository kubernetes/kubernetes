// +build fixtures

package tokens

import (
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/rackspace/gophercloud/openstack/identity/v2/tenants"
	th "github.com/rackspace/gophercloud/testhelper"
)

// ExpectedToken is the token that should be parsed from TokenCreationResponse.
var ExpectedToken = &Token{
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
var ExpectedServiceCatalog = &ServiceCatalog{
	Entries: []CatalogEntry{
		CatalogEntry{
			Name: "inscrutablewalrus",
			Type: "something",
			Endpoints: []Endpoint{
				Endpoint{
					PublicURL: "http://something0:1234/v2/",
					Region:    "region0",
				},
				Endpoint{
					PublicURL: "http://something1:1234/v2/",
					Region:    "region1",
				},
			},
		},
		CatalogEntry{
			Name: "arbitrarypenguin",
			Type: "else",
			Endpoints: []Endpoint{
				Endpoint{
					PublicURL: "http://else0:4321/v3/",
					Region:    "region0",
				},
			},
		},
	},
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

// IsSuccessful ensures that a CreateResult was successful and contains the correct token and
// service catalog.
func IsSuccessful(t *testing.T, result CreateResult) {
	token, err := result.ExtractToken()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, ExpectedToken, token)

	serviceCatalog, err := result.ExtractServiceCatalog()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, ExpectedServiceCatalog, serviceCatalog)
}
