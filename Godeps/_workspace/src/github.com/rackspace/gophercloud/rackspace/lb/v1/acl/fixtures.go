// +build fixtures

package acl

import (
	"fmt"
	"net/http"
	"strconv"
	"testing"

	th "github.com/rackspace/gophercloud/testhelper"
	fake "github.com/rackspace/gophercloud/testhelper/client"
)

func _rootURL(lbID int) string {
	return "/loadbalancers/" + strconv.Itoa(lbID) + "/accesslist"
}

func mockListResponse(t *testing.T, id int) {
	th.Mux.HandleFunc(_rootURL(id), func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
  "accessList": [
    {
      "address": "206.160.163.21",
      "id": 21,
      "type": "DENY"
    },
    {
      "address": "206.160.163.22",
      "id": 22,
      "type": "DENY"
    },
    {
      "address": "206.160.163.23",
      "id": 23,
      "type": "DENY"
    },
    {
      "address": "206.160.163.24",
      "id": 24,
      "type": "DENY"
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
  "accessList": [
    {
      "address": "206.160.163.21",
      "type": "DENY"
    },
    {
      "address": "206.160.165.11",
      "type": "DENY"
    }
  ]
}
    `)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusAccepted)
	})
}

func mockDeleteAllResponse(t *testing.T, lbID int) {
	th.Mux.HandleFunc(_rootURL(lbID), func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusAccepted)
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

func mockDeleteResponse(t *testing.T, lbID, networkID int) {
	th.Mux.HandleFunc(_rootURL(lbID)+"/"+strconv.Itoa(networkID), func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusAccepted)
	})
}
