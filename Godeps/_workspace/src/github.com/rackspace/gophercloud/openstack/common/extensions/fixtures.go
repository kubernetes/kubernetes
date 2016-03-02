// +build fixtures

package extensions

import (
	"fmt"
	"net/http"
	"testing"

	th "github.com/rackspace/gophercloud/testhelper"
	"github.com/rackspace/gophercloud/testhelper/client"
)

// ListOutput provides a single page of Extension results.
const ListOutput = `
{
	"extensions": [
		{
			"updated": "2013-01-20T00:00:00-00:00",
			"name": "Neutron Service Type Management",
			"links": [],
			"namespace": "http://docs.openstack.org/ext/neutron/service-type/api/v1.0",
			"alias": "service-type",
			"description": "API for retrieving service providers for Neutron advanced services"
		}
	]
}`

// GetOutput provides a single Extension result.
const GetOutput = `
{
	"extension": {
		"updated": "2013-02-03T10:00:00-00:00",
		"name": "agent",
		"links": [],
		"namespace": "http://docs.openstack.org/ext/agent/api/v2.0",
		"alias": "agent",
		"description": "The agent management extension."
	}
}
`

// ListedExtension is the Extension that should be parsed from ListOutput.
var ListedExtension = Extension{
	Updated:     "2013-01-20T00:00:00-00:00",
	Name:        "Neutron Service Type Management",
	Links:       []interface{}{},
	Namespace:   "http://docs.openstack.org/ext/neutron/service-type/api/v1.0",
	Alias:       "service-type",
	Description: "API for retrieving service providers for Neutron advanced services",
}

// ExpectedExtensions is a slice containing the Extension that should be parsed from ListOutput.
var ExpectedExtensions = []Extension{ListedExtension}

// SingleExtension is the Extension that should be parsed from GetOutput.
var SingleExtension = &Extension{
	Updated:     "2013-02-03T10:00:00-00:00",
	Name:        "agent",
	Links:       []interface{}{},
	Namespace:   "http://docs.openstack.org/ext/agent/api/v2.0",
	Alias:       "agent",
	Description: "The agent management extension.",
}

// HandleListExtensionsSuccessfully creates an HTTP handler at `/extensions` on the test handler
// mux that response with a list containing a single tenant.
func HandleListExtensionsSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/extensions", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")

		fmt.Fprintf(w, ListOutput)
	})
}

// HandleGetExtensionSuccessfully creates an HTTP handler at `/extensions/agent` that responds with
// a JSON payload corresponding to SingleExtension.
func HandleGetExtensionSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/extensions/agent", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, GetOutput)
	})
}
