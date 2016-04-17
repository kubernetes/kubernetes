// +build fixtures

package serviceassets

import (
	"net/http"
	"testing"

	th "github.com/rackspace/gophercloud/testhelper"
	fake "github.com/rackspace/gophercloud/testhelper/client"
)

// HandleDeleteCDNAssetSuccessfully creates an HTTP handler at `/services/{id}/assets` on the test handler mux
// that responds with a `Delete` response.
func HandleDeleteCDNAssetSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/services/96737ae3-cfc1-4c72-be88-5d0e7cc9a3f0/assets", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusAccepted)
	})
}
