package testing

import (
	"fmt"
	"net/http"
	"testing"

	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

const shareEndpoint = "/shares"

var createRequest = `{
               "share": {
                       "name": "my_test_share",
                       "size": 1,
                       "share_proto": "NFS",
                       "snapshot_id": "70bfbebc-d3ff-4528-8bbb-58422daa280b"
               }
       }`

var createResponse = `{
       "itemNotFound": {
               "code": 404,
               "message": "ShareSnapshotNotFound: Snapshot 70bfbebc-d3ff-4528-8bbb-58422daa280b could not be found."
       }
}`

// MockCreateResponse creates a mock response
func MockCreateResponse(t *testing.T) {
	th.Mux.HandleFunc(shareEndpoint, func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, createRequest)
		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		fmt.Fprintf(w, createResponse)
	})
}
