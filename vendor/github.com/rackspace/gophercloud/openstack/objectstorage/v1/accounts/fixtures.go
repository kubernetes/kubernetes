// +build fixtures

package accounts

import (
	"net/http"
	"testing"

	th "github.com/rackspace/gophercloud/testhelper"
	fake "github.com/rackspace/gophercloud/testhelper/client"
)

// HandleGetAccountSuccessfully creates an HTTP handler at `/` on the test handler mux that
// responds with a `Get` response.
func HandleGetAccountSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "HEAD")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Set("X-Account-Container-Count", "2")
		w.Header().Set("X-Account-Bytes-Used", "14")
		w.Header().Set("X-Account-Meta-Subject", "books")

		w.WriteHeader(http.StatusNoContent)
	})
}

// HandleUpdateAccountSuccessfully creates an HTTP handler at `/` on the test handler mux that
// responds with a `Update` response.
func HandleUpdateAccountSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "X-Account-Meta-Gophercloud-Test", "accounts")

		w.WriteHeader(http.StatusNoContent)
	})
}
