package rackspace

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/rackspace/gophercloud"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestAuthenticatedClientV2(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/tokens", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, `
      {
        "access": {
          "token": {
            "id": "01234567890",
            "expires": "2014-10-01T10:00:00.000000Z"
          },
          "serviceCatalog": []
        }
      }
    `)
	})

	options := gophercloud.AuthOptions{
		Username:         "me",
		APIKey:           "09876543210",
		IdentityEndpoint: th.Endpoint() + "v2.0/",
	}
	client, err := AuthenticatedClient(options)
	th.AssertNoErr(t, err)
	th.CheckEquals(t, "01234567890", client.TokenID)
}
