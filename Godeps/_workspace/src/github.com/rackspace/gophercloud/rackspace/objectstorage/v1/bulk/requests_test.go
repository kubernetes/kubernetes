package bulk

import (
	"fmt"
	"net/http"
	"testing"

	th "github.com/rackspace/gophercloud/testhelper"
	fake "github.com/rackspace/gophercloud/testhelper/client"
)

func TestBulkDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	th.Mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.AssertEquals(t, r.URL.RawQuery, "bulk-delete")

		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `
      {
        "Number Not Found": 1,
        "Response Status": "200 OK",
        "Errors": [],
        "Number Deleted": 1,
        "Response Body": ""
      }
    `)
	})

	options := DeleteOpts{"gophercloud-testcontainer1", "gophercloud-testcontainer2"}
	actual, err := Delete(fake.ServiceClient(), options).ExtractBody()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, actual.NumberDeleted, 1)
}
