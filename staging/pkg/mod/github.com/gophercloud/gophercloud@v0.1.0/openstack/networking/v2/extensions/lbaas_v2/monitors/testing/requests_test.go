package testing

import (
	"testing"

	fake "github.com/gophercloud/gophercloud/openstack/networking/v2/common"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/lbaas_v2/monitors"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestListHealthmonitors(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleHealthmonitorListSuccessfully(t)

	pages := 0
	err := monitors.List(fake.ServiceClient(), monitors.ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		pages++

		actual, err := monitors.ExtractMonitors(page)
		if err != nil {
			return false, err
		}

		if len(actual) != 2 {
			t.Fatalf("Expected 2 healthmonitors, got %d", len(actual))
		}
		th.CheckDeepEquals(t, HealthmonitorWeb, actual[0])
		th.CheckDeepEquals(t, HealthmonitorDb, actual[1])

		return true, nil
	})

	th.AssertNoErr(t, err)

	if pages != 1 {
		t.Errorf("Expected 1 page, saw %d", pages)
	}
}

func TestListAllHealthmonitors(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleHealthmonitorListSuccessfully(t)

	allPages, err := monitors.List(fake.ServiceClient(), monitors.ListOpts{}).AllPages()
	th.AssertNoErr(t, err)
	actual, err := monitors.ExtractMonitors(allPages)
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, HealthmonitorWeb, actual[0])
	th.CheckDeepEquals(t, HealthmonitorDb, actual[1])
}

func TestCreateHealthmonitor(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleHealthmonitorCreationSuccessfully(t, SingleHealthmonitorBody)

	actual, err := monitors.Create(fake.ServiceClient(), monitors.CreateOpts{
		Type:          "HTTP",
		Name:          "db",
		PoolID:        "84f1b61f-58c4-45bf-a8a9-2dafb9e5214d",
		TenantID:      "453105b9-1754-413f-aab1-55f1af620750",
		Delay:         20,
		Timeout:       10,
		MaxRetries:    5,
		URLPath:       "/check",
		ExpectedCodes: "200-299",
	}).Extract()
	th.AssertNoErr(t, err)

	th.CheckDeepEquals(t, HealthmonitorDb, *actual)
}

func TestRequiredCreateOpts(t *testing.T) {
	res := monitors.Create(fake.ServiceClient(), monitors.CreateOpts{})
	if res.Err == nil {
		t.Fatalf("Expected error, got none")
	}
	res = monitors.Create(fake.ServiceClient(), monitors.CreateOpts{Type: monitors.TypeHTTP})
	if res.Err == nil {
		t.Fatalf("Expected error, got none")
	}
}

func TestGetHealthmonitor(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleHealthmonitorGetSuccessfully(t)

	client := fake.ServiceClient()
	actual, err := monitors.Get(client, "5d4b5228-33b0-4e60-b225-9b727c1a20e7").Extract()
	if err != nil {
		t.Fatalf("Unexpected Get error: %v", err)
	}

	th.CheckDeepEquals(t, HealthmonitorDb, *actual)
}

func TestDeleteHealthmonitor(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleHealthmonitorDeletionSuccessfully(t)

	res := monitors.Delete(fake.ServiceClient(), "5d4b5228-33b0-4e60-b225-9b727c1a20e7")
	th.AssertNoErr(t, res.Err)
}

func TestUpdateHealthmonitor(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleHealthmonitorUpdateSuccessfully(t)

	client := fake.ServiceClient()
	name := "NewHealthmonitorName"
	actual, err := monitors.Update(client, "5d4b5228-33b0-4e60-b225-9b727c1a20e7", monitors.UpdateOpts{
		Name:          &name,
		Delay:         3,
		Timeout:       20,
		MaxRetries:    10,
		URLPath:       "/another_check",
		ExpectedCodes: "301",
	}).Extract()
	if err != nil {
		t.Fatalf("Unexpected Update error: %v", err)
	}

	th.CheckDeepEquals(t, HealthmonitorUpdated, *actual)
}

func TestDelayMustBeGreaterOrEqualThanTimeout(t *testing.T) {
	_, err := monitors.Create(fake.ServiceClient(), monitors.CreateOpts{
		Type:          "HTTP",
		PoolID:        "d459f7d8-c6ee-439d-8713-d3fc08aeed8d",
		Delay:         1,
		Timeout:       10,
		MaxRetries:    5,
		URLPath:       "/check",
		ExpectedCodes: "200-299",
	}).Extract()

	if err == nil {
		t.Fatalf("Expected error, got none")
	}

	_, err = monitors.Update(fake.ServiceClient(), "453105b9-1754-413f-aab1-55f1af620750", monitors.UpdateOpts{
		Delay:   1,
		Timeout: 10,
	}).Extract()

	if err == nil {
		t.Fatalf("Expected error, got none")
	}
}
