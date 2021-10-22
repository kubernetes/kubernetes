package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/clustering/v1/profiletypes"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

func TestListProfileTypes(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleList1Successfully(t)

	pageCount := 0
	err := profiletypes.List(fake.ServiceClient()).EachPage(func(page pagination.Page) (bool, error) {
		pageCount++
		actual, err := profiletypes.ExtractProfileTypes(page)
		th.AssertNoErr(t, err)

		th.AssertDeepEquals(t, ExpectedProfileTypes, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)

	if pageCount != 1 {
		t.Errorf("Expected 1 page, got %d", pageCount)
	}
}

func TestGetProfileType10(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleGet1Successfully(t, ExpectedProfileType1.Name)

	actual, err := profiletypes.Get(fake.ServiceClient(), ExpectedProfileType1.Name).Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, ExpectedProfileType1, *actual)
}

func TestGetProfileType15(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleGet15Successfully(t, ExpectedProfileType15.Name)

	actual, err := profiletypes.Get(fake.ServiceClient(), ExpectedProfileType15.Name).Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, ExpectedProfileType15, *actual)
}

func TestListProfileTypesOps(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleListOpsSuccessfully(t)

	allPages, err := profiletypes.ListOps(fake.ServiceClient(), ProfileTypeName).AllPages()
	th.AssertNoErr(t, err)

	allPolicyTypes, err := profiletypes.ExtractOps(allPages)
	th.AssertNoErr(t, err)

	for k, v := range allPolicyTypes {
		tools.PrintResource(t, k)
		tools.PrintResource(t, v)
	}
}
