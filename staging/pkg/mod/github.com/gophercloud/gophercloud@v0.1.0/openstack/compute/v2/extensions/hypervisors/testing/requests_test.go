package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/hypervisors"
	"github.com/gophercloud/gophercloud/pagination"
	"github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

func TestListHypervisorsPre253(t *testing.T) {
	testhelper.SetupHTTP()
	defer testhelper.TeardownHTTP()
	HandleHypervisorListPre253Successfully(t)

	pages := 0
	err := hypervisors.List(client.ServiceClient()).EachPage(func(page pagination.Page) (bool, error) {
		pages++

		actual, err := hypervisors.ExtractHypervisors(page)
		if err != nil {
			return false, err
		}

		if len(actual) != 2 {
			t.Fatalf("Expected 2 hypervisors, got %d", len(actual))
		}
		testhelper.CheckDeepEquals(t, HypervisorFakePre253, actual[0])
		testhelper.CheckDeepEquals(t, HypervisorFakePre253, actual[1])

		return true, nil
	})

	testhelper.AssertNoErr(t, err)

	if pages != 1 {
		t.Errorf("Expected 1 page, saw %d", pages)
	}
}

func TestListAllHypervisorsPre253(t *testing.T) {
	testhelper.SetupHTTP()
	defer testhelper.TeardownHTTP()
	HandleHypervisorListPre253Successfully(t)

	allPages, err := hypervisors.List(client.ServiceClient()).AllPages()
	testhelper.AssertNoErr(t, err)
	actual, err := hypervisors.ExtractHypervisors(allPages)
	testhelper.AssertNoErr(t, err)
	testhelper.CheckDeepEquals(t, HypervisorFakePre253, actual[0])
	testhelper.CheckDeepEquals(t, HypervisorFakePre253, actual[1])
}

func TestListHypervisors(t *testing.T) {
	testhelper.SetupHTTP()
	defer testhelper.TeardownHTTP()
	HandleHypervisorListSuccessfully(t)

	pages := 0
	err := hypervisors.List(client.ServiceClient()).EachPage(func(page pagination.Page) (bool, error) {
		pages++

		actual, err := hypervisors.ExtractHypervisors(page)
		if err != nil {
			return false, err
		}

		if len(actual) != 2 {
			t.Fatalf("Expected 2 hypervisors, got %d", len(actual))
		}
		testhelper.CheckDeepEquals(t, HypervisorFake, actual[0])
		testhelper.CheckDeepEquals(t, HypervisorFake, actual[1])

		return true, nil
	})

	testhelper.AssertNoErr(t, err)

	if pages != 1 {
		t.Errorf("Expected 1 page, saw %d", pages)
	}
}

func TestListAllHypervisors(t *testing.T) {
	testhelper.SetupHTTP()
	defer testhelper.TeardownHTTP()
	HandleHypervisorListSuccessfully(t)

	allPages, err := hypervisors.List(client.ServiceClient()).AllPages()
	testhelper.AssertNoErr(t, err)
	actual, err := hypervisors.ExtractHypervisors(allPages)
	testhelper.AssertNoErr(t, err)
	testhelper.CheckDeepEquals(t, HypervisorFake, actual[0])
	testhelper.CheckDeepEquals(t, HypervisorFake, actual[1])
}

func TestHypervisorsStatistics(t *testing.T) {
	testhelper.SetupHTTP()
	defer testhelper.TeardownHTTP()
	HandleHypervisorsStatisticsSuccessfully(t)

	expected := HypervisorsStatisticsExpected

	actual, err := hypervisors.GetStatistics(client.ServiceClient()).Extract()
	testhelper.AssertNoErr(t, err)
	testhelper.CheckDeepEquals(t, &expected, actual)
}

func TestGetHypervisor(t *testing.T) {
	testhelper.SetupHTTP()
	defer testhelper.TeardownHTTP()
	HandleHypervisorGetSuccessfully(t)

	expected := HypervisorFake

	actual, err := hypervisors.Get(client.ServiceClient(), expected.ID).Extract()
	testhelper.AssertNoErr(t, err)
	testhelper.CheckDeepEquals(t, &expected, actual)
}

func TestGetHypervisorEmptyCPUInfo(t *testing.T) {
	testhelper.SetupHTTP()
	defer testhelper.TeardownHTTP()
	HandleHypervisorGetEmptyCPUInfoSuccessfully(t)

	expected := HypervisorEmptyCPUInfo

	actual, err := hypervisors.Get(client.ServiceClient(), expected.ID).Extract()
	testhelper.AssertNoErr(t, err)
	testhelper.CheckDeepEquals(t, &expected, actual)
}

func TestHypervisorsUptime(t *testing.T) {
	testhelper.SetupHTTP()
	defer testhelper.TeardownHTTP()
	HandleHypervisorUptimeSuccessfully(t)

	expected := HypervisorUptimeExpected

	actual, err := hypervisors.GetUptime(client.ServiceClient(), HypervisorFake.ID).Extract()
	testhelper.AssertNoErr(t, err)
	testhelper.CheckDeepEquals(t, &expected, actual)
}
