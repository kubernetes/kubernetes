package testing

import (
	"fmt"
	"net/http"
	"testing"

	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

// ListOutput provides a single Extension result. It differs from the delegated implementation
// by the introduction of an intermediate "values" member.
const ListOutput = `
{
	"extensions": {
		"values": [
			{
				"updated": "2013-01-20T00:00:00-00:00",
				"name": "Neutron Service Type Management",
				"links": [],
				"namespace": "http://docs.openstack.org/ext/neutron/service-type/api/v1.0",
				"alias": "service-type",
				"description": "API for retrieving service providers for Neutron advanced services"
			}
		]
	}
}
`

// HandleListExtensionsSuccessfully creates an HTTP handler that returns ListOutput for a List
// call.
func HandleListExtensionsSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/extensions", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")

		fmt.Fprintf(w, `
{
  "extensions": {
    "values": [
      {
        "updated": "2013-01-20T00:00:00-00:00",
        "name": "Neutron Service Type Management",
        "links": [],
        "namespace": "http://docs.openstack.org/ext/neutron/service-type/api/v1.0",
        "alias": "service-type",
        "description": "API for retrieving service providers for Neutron advanced services"
      }
    ]
  }
}
    `)
	})

}
