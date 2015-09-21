package cloudnetworks

import (
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
	fake "github.com/rackspace/gophercloud/testhelper/client"
)

func TestListCloudNetworks(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	th.Mux.HandleFunc("/cloud_networks", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")

		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `[{
      "cidr": "192.168.100.0/24",
      "created": "2014-05-25T01:23:42Z",
      "id": "07426958-1ebf-4c38-b032-d456820ca21a",
      "name": "RC-CLOUD",
      "updated": "2014-05-25T02:28:44Z"
    }]`)
	})

	expected := []CloudNetwork{
		CloudNetwork{
			CIDR:      "192.168.100.0/24",
			CreatedAt: time.Date(2014, 5, 25, 1, 23, 42, 0, time.UTC),
			ID:        "07426958-1ebf-4c38-b032-d456820ca21a",
			Name:      "RC-CLOUD",
			UpdatedAt: time.Date(2014, 5, 25, 2, 28, 44, 0, time.UTC),
		},
	}

	count := 0
	err := List(fake.ServiceClient()).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := ExtractCloudNetworks(page)
		th.AssertNoErr(t, err)

		th.CheckDeepEquals(t, expected, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.CheckEquals(t, count, 1)
}

func TestGetCloudNetwork(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	th.Mux.HandleFunc("/cloud_networks/07426958-1ebf-4c38-b032-d456820ca21a", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `{
      "cidr": "192.168.100.0/24",
      "created": "2014-05-25T01:23:42Z",
      "id": "07426958-1ebf-4c38-b032-d456820ca21a",
      "name": "RC-CLOUD",
      "updated": "2014-05-25T02:28:44Z"
    }`)
	})

	expected := &CloudNetwork{
		CIDR:      "192.168.100.0/24",
		CreatedAt: time.Date(2014, 5, 25, 1, 23, 42, 0, time.UTC),
		ID:        "07426958-1ebf-4c38-b032-d456820ca21a",
		Name:      "RC-CLOUD",
		UpdatedAt: time.Date(2014, 5, 25, 2, 28, 44, 0, time.UTC),
	}

	actual, err := Get(fake.ServiceClient(), "07426958-1ebf-4c38-b032-d456820ca21a").Extract()
	th.AssertNoErr(t, err)

	th.AssertDeepEquals(t, expected, actual)
}
