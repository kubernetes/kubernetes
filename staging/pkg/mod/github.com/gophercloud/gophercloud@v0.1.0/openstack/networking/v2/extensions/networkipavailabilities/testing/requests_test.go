package testing

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/gophercloud/gophercloud"
	fake "github.com/gophercloud/gophercloud/openstack/networking/v2/common"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/networkipavailabilities"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/network-ip-availabilities", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, NetworkIPAvailabilityListResult)
	})

	count := 0

	err := networkipavailabilities.List(fake.ServiceClient(), networkipavailabilities.ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := networkipavailabilities.ExtractNetworkIPAvailabilities(page)
		if err != nil {
			t.Errorf("Failed to extract network IP availabilities: %v", err)
			return false, nil
		}

		expected := []networkipavailabilities.NetworkIPAvailability{
			NetworkIPAvailability1,
			NetworkIPAvailability2,
		}

		th.CheckDeepEquals(t, expected, actual)

		return true, nil
	})

	th.AssertNoErr(t, err)

	if count != 1 {
		t.Errorf("Expected 1 page, got %d", count)
	}
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/network-ip-availabilities/cf11ab78-2302-49fa-870f-851a08c7afb8", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, NetworkIPAvailabilityGetResult)
	})

	s, err := networkipavailabilities.Get(fake.ServiceClient(), "cf11ab78-2302-49fa-870f-851a08c7afb8").Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, s.NetworkID, "cf11ab78-2302-49fa-870f-851a08c7afb8")
	th.AssertEquals(t, s.NetworkName, "public")
	th.AssertEquals(t, s.ProjectID, "424e7cf0243c468ca61732ba45973b3e")
	th.AssertEquals(t, s.TenantID, "424e7cf0243c468ca61732ba45973b3e")
	th.AssertEquals(t, s.TotalIPs, "253")
	th.AssertEquals(t, s.UsedIPs, "3")
	th.AssertDeepEquals(t, s.SubnetIPAvailabilities, []networkipavailabilities.SubnetIPAvailability{
		{
			SubnetID:   "4afe6e5f-9649-40db-b18f-64c7ead942bd",
			SubnetName: "public-subnet",
			CIDR:       "203.0.113.0/24",
			IPVersion:  int(gophercloud.IPv4),
			TotalIPs:   "253",
			UsedIPs:    "3",
		},
	})
}
