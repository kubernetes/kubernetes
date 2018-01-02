package testing

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/gophercloud/gophercloud/openstack/identity/v3/domains"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

// ListOutput provides a single page of Domain results.
const ListOutput = `
{
    "links": {
        "next": null,
        "previous": null,
        "self": "http://example.com/identity/v3/domains"
    },
    "domains": [
        {
            "enabled": true,
            "id": "2844b2a08be147a08ef58317d6471f1f",
            "links": {
                "self": "http://example.com/identity/v3/domains/2844b2a08be147a08ef58317d6471f1f"
            },
            "name": "domain one",
            "description": "some description"
        },
        {
            "enabled": true,
            "id": "9fe1d3",
            "links": {
                "self": "https://example.com/identity/v3/domains/9fe1d3"
            },
            "name": "domain two"
        }
    ]
}
`

// GetOutput provides a Get result.
const GetOutput = `
{
    "domain": {
        "enabled": true,
        "id": "9fe1d3",
        "links": {
            "self": "https://example.com/identity/v3/domains/9fe1d3"
        },
        "name": "domain two"
    }
}
`

// CreateRequest provides the input to a Create request.
const CreateRequest = `
{
    "domain": {
        "name": "domain two"
    }
}
`

// UpdateRequest provides the input to as Update request.
const UpdateRequest = `
{
    "domain": {
        "description": "Staging Domain"
    }
}
`

// UpdateOutput provides an update result.
const UpdateOutput = `
{
    "domain": {
		"enabled": true,
        "id": "9fe1d3",
        "links": {
            "self": "https://example.com/identity/v3/domains/9fe1d3"
        },
        "name": "domain two",
        "description": "Staging Domain"
    }
}
`

// FirstDomain is the first domain in the List request.
var FirstDomain = domains.Domain{
	Enabled: true,
	ID:      "2844b2a08be147a08ef58317d6471f1f",
	Links: map[string]interface{}{
		"self": "http://example.com/identity/v3/domains/2844b2a08be147a08ef58317d6471f1f",
	},
	Name:        "domain one",
	Description: "some description",
}

// SecondDomain is the second domain in the List request.
var SecondDomain = domains.Domain{
	Enabled: true,
	ID:      "9fe1d3",
	Links: map[string]interface{}{
		"self": "https://example.com/identity/v3/domains/9fe1d3",
	},
	Name: "domain two",
}

// SecondDomainUpdated is how SecondDomain should look after an Update.
var SecondDomainUpdated = domains.Domain{
	Enabled: true,
	ID:      "9fe1d3",
	Links: map[string]interface{}{
		"self": "https://example.com/identity/v3/domains/9fe1d3",
	},
	Name:        "domain two",
	Description: "Staging Domain",
}

// ExpectedDomainsSlice is the slice of domains expected to be returned from ListOutput.
var ExpectedDomainsSlice = []domains.Domain{FirstDomain, SecondDomain}

// HandleListDomainsSuccessfully creates an HTTP handler at `/domains` on the
// test handler mux that responds with a list of two domains.
func HandleListDomainsSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/domains", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, ListOutput)
	})
}

// HandleGetDomainSuccessfully creates an HTTP handler at `/domains` on the
// test handler mux that responds with a single domain.
func HandleGetDomainSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/domains/9fe1d3", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, GetOutput)
	})
}

// HandleCreateDomainSuccessfully creates an HTTP handler at `/domains` on the
// test handler mux that tests domain creation.
func HandleCreateDomainSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/domains", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, CreateRequest)

		w.WriteHeader(http.StatusCreated)
		fmt.Fprintf(w, GetOutput)
	})
}

// HandleDeleteDomainSuccessfully creates an HTTP handler at `/domains` on the
// test handler mux that tests domain deletion.
func HandleDeleteDomainSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/domains/9fe1d3", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.WriteHeader(http.StatusNoContent)
	})
}

// HandleUpdateDomainSuccessfully creates an HTTP handler at `/domains` on the
// test handler mux that tests domain update.
func HandleUpdateDomainSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/domains/9fe1d3", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PATCH")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, UpdateRequest)

		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, UpdateOutput)
	})
}
