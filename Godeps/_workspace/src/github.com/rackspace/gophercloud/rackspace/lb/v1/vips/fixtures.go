package vips

import (
	"fmt"
	"net/http"
	"strconv"
	"testing"

	th "github.com/rackspace/gophercloud/testhelper"
	fake "github.com/rackspace/gophercloud/testhelper/client"
)

func _rootURL(lbID int) string {
	return "/loadbalancers/" + strconv.Itoa(lbID) + "/virtualips"
}

func mockListResponse(t *testing.T, lbID int) {
	th.Mux.HandleFunc(_rootURL(lbID), func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
  "virtualIps": [
    {
      "id": 1000,
      "address": "206.10.10.210",
      "type": "PUBLIC"
    }
  ]
}
  `)
	})
}

func mockCreateResponse(t *testing.T, lbID int) {
	th.Mux.HandleFunc(_rootURL(lbID), func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		th.TestJSONRequest(t, r, `
{
    "type":"PUBLIC",
    "ipVersion":"IPV6"
}
    `)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusAccepted)

		fmt.Fprintf(w, `
{
    "address":"fd24:f480:ce44:91bc:1af2:15ff:0000:0002",
    "id":9000134,
    "type":"PUBLIC",
    "ipVersion":"IPV6"
}
  `)
	})
}

func mockBatchDeleteResponse(t *testing.T, lbID int, ids []int) {
	th.Mux.HandleFunc(_rootURL(lbID), func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		r.ParseForm()

		for k, v := range ids {
			fids := r.Form["id"]
			th.AssertEquals(t, strconv.Itoa(v), fids[k])
		}

		w.WriteHeader(http.StatusAccepted)
	})
}

func mockDeleteResponse(t *testing.T, lbID, vipID int) {
	url := _rootURL(lbID) + "/" + strconv.Itoa(vipID)
	th.Mux.HandleFunc(url, func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusAccepted)
	})
}
