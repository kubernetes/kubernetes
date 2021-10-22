package testing

import (
	"fmt"
	"net/http"
	"testing"

	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

const projectID = "aa5436ab58144c768ca4e9d2e9f5c3b2"
const requestUUID = "req-781e9bdc-4163-46eb-91c9-786c53188bbb"

var CreateResponse = fmt.Sprintf(`
{
   "resource": "Cluster",
   "created_at": "2017-01-17T17:35:48+00:00",
   "updated_at": null,
   "hard_limit": 1,
   "project_id": "%s",
   "id": 26
}`, projectID)

func HandleCreateQuotaSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/quotas", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.Header().Add("X-OpenStack-Request-Id", requestUUID)
		w.WriteHeader(http.StatusCreated)

		fmt.Fprint(w, CreateResponse)
	})
}
