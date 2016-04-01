// +build fixtures

package flavors

import (
	"fmt"
	"net/http"
	"testing"

	th "github.com/rackspace/gophercloud/testhelper"
	fake "github.com/rackspace/gophercloud/testhelper/client"
)

// HandleListCDNFlavorsSuccessfully creates an HTTP handler at `/flavors` on the test handler mux
// that responds with a `List` response.
func HandleListCDNFlavorsSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/flavors", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `
      {
        "flavors": [
            {
                "id": "europe",
                "providers": [
                    {
                        "provider": "Fastly",
                        "links": [
                            {
                                "href": "http://www.fastly.com",
                                "rel": "provider_url"
                            }
                        ]
                    }
                ],
                "links": [
                    {
                        "href": "https://www.poppycdn.io/v1.0/flavors/europe",
                        "rel": "self"
                    }
                ]
            }
        ]
    }
    `)
	})
}

// HandleGetCDNFlavorSuccessfully creates an HTTP handler at `/flavors/{id}` on the test handler mux
// that responds with a `Get` response.
func HandleGetCDNFlavorSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/flavors/asia", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `
      {
          "id" : "asia",
          "providers" : [
              {
                  "provider" : "ChinaCache",
                  "links": [
                      {
                          "href": "http://www.chinacache.com",
                          "rel": "provider_url"
                      }
                  ]
              }
          ],
          "links": [
              {
                  "href": "https://www.poppycdn.io/v1.0/flavors/asia",
                  "rel": "self"
              }
          ]
      }
    `)
	})
}
