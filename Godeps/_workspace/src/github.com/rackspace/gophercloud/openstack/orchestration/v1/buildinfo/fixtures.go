// +build fixtures

package buildinfo

import (
	"fmt"
	"net/http"
	"testing"

	th "github.com/rackspace/gophercloud/testhelper"
	fake "github.com/rackspace/gophercloud/testhelper/client"
)

// GetExpected represents the expected object from a Get request.
var GetExpected = &BuildInfo{
	API: Revision{
		Revision: "2.4.5",
	},
	Engine: Revision{
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
