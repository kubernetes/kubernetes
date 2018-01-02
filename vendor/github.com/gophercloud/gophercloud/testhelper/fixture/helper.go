package fixture

import (
	"fmt"
	"net/http"
	"testing"

	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

func SetupHandler(t *testing.T, url, method, requestBody, responseBody string, status int) {
	th.Mux.HandleFunc(url, func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, method)
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		if requestBody != "" {
			th.TestJSONRequest(t, r, requestBody)
		}

		if responseBody != "" {
			w.Header().Add("Content-Type", "application/json")
		}

		w.WriteHeader(status)

		if responseBody != "" {
			fmt.Fprintf(w, responseBody)
		}
	})
}
