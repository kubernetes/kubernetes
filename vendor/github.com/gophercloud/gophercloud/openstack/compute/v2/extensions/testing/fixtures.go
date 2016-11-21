package testing

import (
	"fmt"
	"net/http"
	"testing"

	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

func HandleListExtensionsSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/extensions", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")

		fmt.Fprintf(w, `
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
}
			`)
	})
}

func HandleGetExtensionsSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/extensions/agent", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
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
		`)
	})
}
