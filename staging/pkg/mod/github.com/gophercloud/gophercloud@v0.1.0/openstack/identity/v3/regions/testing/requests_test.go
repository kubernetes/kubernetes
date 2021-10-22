package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud/openstack/identity/v3/regions"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

func TestListRegions(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListRegionsSuccessfully(t)

	count := 0
	err := regions.List(client.ServiceClient(), nil).EachPage(func(page pagination.Page) (bool, error) {
		count++

		actual, err := regions.ExtractRegions(page)
		th.AssertNoErr(t, err)

		th.CheckDeepEquals(t, ExpectedRegionsSlice, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.CheckEquals(t, count, 1)
}

func TestListRegionsAllPages(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListRegionsSuccessfully(t)

	allPages, err := regions.List(client.ServiceClient(), nil).AllPages()
	th.AssertNoErr(t, err)
	actual, err := regions.ExtractRegions(allPages)
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, ExpectedRegionsSlice, actual)
	th.AssertEquals(t, ExpectedRegionsSlice[1].Extra["email"], "westsupport@example.com")
}

func TestGetRegion(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleGetRegionSuccessfully(t)

	actual, err := regions.Get(client.ServiceClient(), "RegionOne-West").Extract()

	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, SecondRegion, *actual)
}

func TestCreateRegion(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleCreateRegionSuccessfully(t)

	createOpts := regions.CreateOpts{
		ID:          "RegionOne-West",
		Description: "West sub-region of RegionOne",
		Extra: map[string]interface{}{
			"email": "westsupport@example.com",
		},
		ParentRegionID: "RegionOne",
	}

	actual, err := regions.Create(client.ServiceClient(), createOpts).Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, SecondRegion, *actual)
}

func TestUpdateRegion(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleUpdateRegionSuccessfully(t)

	var description = "First West sub-region of RegionOne"
	updateOpts := regions.UpdateOpts{
		Description: &description,
		/*
			// Due to a bug in Keystone, the Extra column of the Region table
			// is not updatable, see: https://bugs.launchpad.net/keystone/+bug/1729933
			// The following lines should be uncommented once the fix is merged.

			Extra: map[string]interface{}{
				"email": "1stwestsupport@example.com",
			},
		*/
	}

	actual, err := regions.Update(client.ServiceClient(), "RegionOne-West", updateOpts).Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, SecondRegionUpdated, *actual)
}

func TestDeleteRegion(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleDeleteRegionSuccessfully(t)

	res := regions.Delete(client.ServiceClient(), "RegionOne-West")
	th.AssertNoErr(t, res.Err)
}
