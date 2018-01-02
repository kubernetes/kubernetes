package testing

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/gophercloud/gophercloud/openstack/identity/v3/services"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

// ListOutput provides a single page of Service results.
const ListOutput = `
{
    "links": {
        "next": null,
        "previous": null
    },
    "services": [
        {
            "id": "1234",
            "links": {
                "self": "https://example.com/identity/v3/services/1234"
            },
            "type": "identity",
            "enabled": false,
            "extra": {
                "name": "service-one",
                "description": "Service One"
            }
        },
        {
            "id": "9876",
            "links": {
                "self": "https://example.com/identity/v3/services/9876"
            },
            "type": "compute",
            "enabled": false,
            "extra": {
                "name": "service-two",
                "description": "Service Two",
                "email": "service@example.com"
            }
        }
    ]
}
`

// GetOutput provides a Get result.
const GetOutput = `
{
    "service": {
        "id": "9876",
        "links": {
            "self": "https://example.com/identity/v3/services/9876"
        },
        "type": "compute",
        "enabled": false,
        "extra": {
            "name": "service-two",
            "description": "Service Two",
            "email": "service@example.com"
        }
    }
}
`

// CreateRequest provides the input to a Create request.
const CreateRequest = `
{
    "service": {
        "description": "Service Two",
        "email": "service@example.com",
        "name": "service-two",
        "type": "compute"
    }
}
`

// UpdateRequest provides the input to as Update request.
const UpdateRequest = `
{
    "service": {
        "type": "compute2",
        "description": "Service Two Updated"
    }
}
`

// UpdateOutput provides an update result.
const UpdateOutput = `
{
    "service": {
        "id": "9876",
        "links": {
            "self": "https://example.com/identity/v3/services/9876"
        },
        "type": "compute2",
        "enabled": false,
        "extra": {
            "name": "service-two",
            "description": "Service Two Updated",
            "email": "service@example.com"
        }
    }
}
`

// FirstService is the first service in the List request.
var FirstService = services.Service{
	ID: "1234",
	Links: map[string]interface{}{
		"self": "https://example.com/identity/v3/services/1234",
	},
	Type:    "identity",
	Enabled: false,
	Extra: map[string]interface{}{
		"name":        "service-one",
		"description": "Service One",
	},
}

// SecondService is the second service in the List request.
var SecondService = services.Service{
	ID: "9876",
	Links: map[string]interface{}{
		"self": "https://example.com/identity/v3/services/9876",
	},
	Type:    "compute",
	Enabled: false,
	Extra: map[string]interface{}{
		"name":        "service-two",
		"description": "Service Two",
		"email":       "service@example.com",
	},
}

// SecondServiceUpdated is the SecondService should look after an Update.
var SecondServiceUpdated = services.Service{
	ID: "9876",
	Links: map[string]interface{}{
		"self": "https://example.com/identity/v3/services/9876",
	},
	Type:    "compute2",
	Enabled: false,
	Extra: map[string]interface{}{
		"name":        "service-two",
		"description": "Service Two Updated",
		"email":       "service@example.com",
	},
}

// ExpectedServicesSlice is the slice of services to be returned from ListOutput.
var ExpectedServicesSlice = []services.Service{FirstService, SecondService}

// HandleListServicesSuccessfully creates an HTTP handler at `/services` on the
// test handler mux that responds with a list of two services.
func HandleListServicesSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/services", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, ListOutput)
	})
}

// HandleGetServiceSuccessfully creates an HTTP handler at `/services` on the
// test handler mux that responds with a single service.
func HandleGetServiceSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/services/9876", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, GetOutput)
	})
}

// HandleCreateServiceSuccessfully creates an HTTP handler at `/services` on the
// test handler mux that tests service creation.
func HandleCreateServiceSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/services", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, CreateRequest)

		w.WriteHeader(http.StatusCreated)
		fmt.Fprintf(w, GetOutput)
	})
}

// HandleUpdateServiceSuccessfully creates an HTTP handler at `/services` on the
// test handler mux that tests service update.
func HandleUpdateServiceSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/services/9876", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PATCH")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, UpdateRequest)

		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, UpdateOutput)
	})
}
