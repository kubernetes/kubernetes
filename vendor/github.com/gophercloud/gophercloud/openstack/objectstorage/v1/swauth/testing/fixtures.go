package testing

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/gophercloud/gophercloud/openstack/objectstorage/v1/swauth"
	th "github.com/gophercloud/gophercloud/testhelper"
)

// AuthResult is the expected result of AuthOutput
var AuthResult = swauth.AuthResult{
	Token:      "AUTH_tk6223e6071f8f4299aa334b48015484a1",
	StorageURL: "http://127.0.0.1:8080/v1/AUTH_test",
}

// HandleAuthSuccessfully configures the test server to respond to an Auth request.
func HandleAuthSuccessfully(t *testing.T, authOpts swauth.AuthOpts) {
	th.Mux.HandleFunc("/auth/v1.0", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-User", authOpts.User)
		th.TestHeader(t, r, "X-Auth-Key", authOpts.Key)

		w.Header().Add("X-Auth-Token", AuthResult.Token)
		w.Header().Add("X-Storage-Url", AuthResult.StorageURL)
		fmt.Fprintf(w, "")
	})
}
