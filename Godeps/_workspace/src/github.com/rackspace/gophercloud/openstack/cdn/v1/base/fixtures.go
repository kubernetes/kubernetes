package base

import (
	"fmt"
	"net/http"
	"testing"

	th "github.com/rackspace/gophercloud/testhelper"
	fake "github.com/rackspace/gophercloud/testhelper/client"
)

// HandleGetSuccessfully creates an HTTP handler at `/` on the test handler mux
// that responds with a `Get` response.
func HandleGetSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `
    {
        "resources": {
            "rel/cdn": {
                "href-template": "services{?marker,limit}",
                "href-vars": {
                    "marker": "param/marker",
                    "limit": "param/limit"
                },
                "hints": {
                    "allow": [
                        "GET"
                    ],
                    "formats": {
                        "application/json": {}
                    }
                }
            }
        }
    }
    `)

	})
}

// HandlePingSuccessfully creates an HTTP handler at `/ping` on the test handler
// mux that responds with a `Ping` response.
func HandlePingSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/ping", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusNoContent)
	})
}
