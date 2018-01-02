package testing

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/gophercloud/gophercloud/openstack/orchestration/v1/buildinfo"
	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

// GetExpected represents the expected object from a Get request.
var GetExpected = &buildinfo.BuildInfo{
	API: buildinfo.Revision{
		Revision: "2.4.5",
	},
	Engine: buildinfo.Revision{
		Revision: "1.2.1",
	},
}

// GetOutput represents the response body from a Get request.
const GetOutput = `
{
  "api": {
    "revision": "2.4.5"
  },
  "engine": {
    "revision": "1.2.1"
  }
}`

// HandleGetSuccessfully creates an HTTP handler at `/build_info`
// on the test handler mux that responds with a `Get` response.
func HandleGetSuccessfully(t *testing.T, output string) {
	th.Mux.HandleFunc("/build_info", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, output)
	})
}
