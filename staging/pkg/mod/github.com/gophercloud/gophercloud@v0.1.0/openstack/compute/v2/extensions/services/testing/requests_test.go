package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/services"
	"github.com/gophercloud/gophercloud/pagination"
	"github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

func TestListServicesPre253(t *testing.T) {
	testhelper.SetupHTTP()
	defer testhelper.TeardownHTTP()
	HandleListPre253Successfully(t)

	pages := 0
	err := services.List(client.ServiceClient()).EachPage(func(page pagination.Page) (bool, error) {
		pages++

		actual, err := services.ExtractServices(page)
		if err != nil {
			return false, err
		}

		if len(actual) != 4 {
			t.Fatalf("Expected 4 services, got %d", len(actual))
		}
		testhelper.CheckDeepEquals(t, FirstFakeServicePre253, actual[0])
		testhelper.CheckDeepEquals(t, SecondFakeServicePre253, actual[1])
		testhelper.CheckDeepEquals(t, ThirdFakeServicePre253, actual[2])
		testhelper.CheckDeepEquals(t, FourthFakeServicePre253, actual[3])

		return true, nil
	})

	testhelper.AssertNoErr(t, err)

	if pages != 1 {
		t.Errorf("Expected 1 page, saw %d", pages)
	}
}

func TestListServices(t *testing.T) {
	testhelper.SetupHTTP()
	defer testhelper.TeardownHTTP()
	HandleListSuccessfully(t)

	pages := 0
	err := services.List(client.ServiceClient()).EachPage(func(page pagination.Page) (bool, error) {
		pages++

		actual, err := services.ExtractServices(page)
		if err != nil {
			return false, err
		}

		if len(actual) != 4 {
			t.Fatalf("Expected 4 services, got %d", len(actual))
		}
		testhelper.CheckDeepEquals(t, FirstFakeService, actual[0])
		testhelper.CheckDeepEquals(t, SecondFakeService, actual[1])
		testhelper.CheckDeepEquals(t, ThirdFakeService, actual[2])
		testhelper.CheckDeepEquals(t, FourthFakeService, actual[3])

		return true, nil
	})

	testhelper.AssertNoErr(t, err)

	if pages != 1 {
		t.Errorf("Expected 1 page, saw %d", pages)
	}
}
