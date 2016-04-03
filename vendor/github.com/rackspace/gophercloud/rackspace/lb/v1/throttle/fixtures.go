package throttle

import (
	"fmt"
	"net/http"
	"strconv"
	"testing"

	th "github.com/rackspace/gophercloud/testhelper"
	fake "github.com/rackspace/gophercloud/testhelper/client"
)

func _rootURL(id int) string {
	return "/loadbalancers/" + strconv.Itoa(id) + "/connectionthrottle"
}

func mockGetResponse(t *testing.T, lbID int) {
	th.Mux.HandleFunc(_rootURL(lbID), func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
  "connectionThrottle": {
    "maxConnections": 100
  }
}
`)
	})
}

func mockCreateResponse(t *testing.T, lbID int) {
	th.Mux.HandleFunc(_rootURL(lbID), func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		th.TestJSONRequest(t, r, `
{
  "connectionThrottle": {
    "maxConnectionRate": 0,
    "maxConnections": 200,
    "minConnections": 0,
    "rateInterval": 0
  }
}
    `)

		w.WriteHeader(http.StatusAccepted)
		fmt.Fprintf(w, `{}`)
	})
}

func mockDeleteResponse(t *testing.T, lbID int) {
	th.Mux.HandleFunc(_rootURL(lbID), func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusAccepted)
	})
}
