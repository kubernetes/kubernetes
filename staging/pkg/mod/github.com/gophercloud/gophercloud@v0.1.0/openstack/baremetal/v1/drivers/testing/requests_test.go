package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud/openstack/baremetal/v1/drivers"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

func TestListDrivers(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListDriversSuccessfully(t)

	pages := 0
	err := drivers.ListDrivers(client.ServiceClient(), drivers.ListDriversOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		pages++

		actual, err := drivers.ExtractDrivers(page)
		if err != nil {
			return false, err
		}

		if len(actual) != 3 {
			t.Fatalf("Expected 3 drivers, got %d", len(actual))
		}

		th.CheckDeepEquals(t, DriverAgentIpmitool, actual[0])
		th.CheckDeepEquals(t, DriverFake, actual[1])
		th.AssertEquals(t, "ipmi", actual[2].Name)

		return true, nil
	})

	th.AssertNoErr(t, err)

	if pages != 1 {
		t.Errorf("Expected 1 page, saw %d", pages)
	}
}

func TestGetDriverDetails(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleGetDriverDetailsSuccessfully(t)

	c := client.ServiceClient()
	actual, err := drivers.GetDriverDetails(c, "ipmi").Extract()
	if err != nil {
		t.Fatalf("Unexpected Get error: %v", err)
	}

	th.CheckDeepEquals(t, DriverIpmi, *actual)
}

func TestGetDriverProperties(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleGetDriverPropertiesSuccessfully(t)

	c := client.ServiceClient()
	actual, err := drivers.GetDriverProperties(c, "agent_ipmitool").Extract()
	if err != nil {
		t.Fatalf("Unexpected Get error: %v", err)
	}

	th.CheckDeepEquals(t, DriverIpmiToolProperties, *actual)
}

func TestGetDriverDiskProperties(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleGetDriverDiskPropertiesSuccessfully(t)

	c := client.ServiceClient()
	actual, err := drivers.GetDriverDiskProperties(c, "agent_ipmitool").Extract()
	if err != nil {
		t.Fatalf("Unexpected Get error: %v", err)
	}

	th.CheckDeepEquals(t, DriverIpmiToolDisk, *actual)
}
