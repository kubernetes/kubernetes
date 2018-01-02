package testing

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/gophercloud/gophercloud/openstack/identity/v2/tenants"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

// ListOutput provides a single page of Tenant results.
const ListOutput = `
{
	"tenants": [
		{
			"id": "1234",
			"name": "Red Team",
			"description": "The team that is red",
			"enabled": true
		},
		{
			"id": "9876",
			"name": "Blue Team",
			"description": "The team that is blue",
			"enabled": false
		}
	]
}
`

// RedTeam is a Tenant fixture.
var RedTeam = tenants.Tenant{
	ID:          "1234",
	Name:        "Red Team",
	Description: "The team that is red",
	Enabled:     true,
}

// BlueTeam is a Tenant fixture.
var BlueTeam = tenants.Tenant{
	ID:          "9876",
	Name:        "Blue Team",
	Description: "The team that is blue",
	Enabled:     false,
}

// ExpectedTenantSlice is the slice of tenants expected to be returned from ListOutput.
var ExpectedTenantSlice = []tenants.Tenant{RedTeam, BlueTeam}

// HandleListTenantsSuccessfully creates an HTTP handler at `/tenants` on the test handler mux that
// responds with a list of two tenants.
func HandleListTenantsSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/tenants", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, ListOutput)
	})
}

func mockCreateTenantResponse(t *testing.T) {
	th.Mux.HandleFunc("/tenants", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		th.TestJSONRequest(t, r, `
{
    "tenant": {
		    "name": "new_tenant",
		    "description": "This is new tenant",
		    "enabled": true
    }
}
	`)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
    "tenant": {
        "name": "new_tenant",
        "description": "This is new tenant",
        "enabled": true,
        "id": "5c62ef576dc7444cbb73b1fe84b97648"
    }
}
`)
	})
}

func mockDeleteTenantResponse(t *testing.T) {
	th.Mux.HandleFunc("/tenants/2466f69cd4714d89a548a68ed97ffcd4", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		w.WriteHeader(http.StatusNoContent)
	})
}

func mockUpdateTenantResponse(t *testing.T) {
	th.Mux.HandleFunc("/tenants/5c62ef576dc7444cbb73b1fe84b97648", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		th.TestJSONRequest(t, r, `
{
    "tenant": {
		    "name": "new_name",
	 	    "description": "This is new name",
		    "enabled": true
    }
}
`)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
		"tenant": {
				"name": "new_name",
				"description": "This is new name",
				"enabled": true,
				"id": "5c62ef576dc7444cbb73b1fe84b97648"
		}
}
`)
	})
}

func mockGetTenantResponse(t *testing.T) {
	th.Mux.HandleFunc("/tenants/5c62ef576dc7444cbb73b1fe84b97648", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
		"tenant": {
				"name": "new_tenant",
				"description": "This is new tenant",
				"enabled": true,
				"id": "5c62ef576dc7444cbb73b1fe84b97648"
		}
}
`)
	})
}
