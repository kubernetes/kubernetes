package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud/openstack/service/vN/resources"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

func TestListResources(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListResourcesSuccessfully(t)

	count := 0
	err := resources.List(client.ServiceClient(), nil).EachPage(func(page pagination.Page) (bool, error) {
		count++

		actual, err := resources.ExtractResources(page)
		th.AssertNoErr(t, err)

		th.AssertDeepEquals(t, ExpectedResourcesSlice, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.AssertEquals(t, count, 1)
}

func TestListResourcesAllPages(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListResourcesSuccessfully(t)

	allPages, err := resources.List(client.ServiceClient(), nil).AllPages()
	th.AssertNoErr(t, err)
	actual, err := resources.ExtractResources(allPages)
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, ExpectedResourcesSlice, actual)
}

func TestGetResource(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleGetResourceSuccessfully(t)

	actual, err := resources.Get(client.ServiceClient(), "9fe1d3").Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, SecondResource, *actual)
}

func TestCreateResource(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleCreateResourceSuccessfully(t)

	createOpts := resources.CreateOpts{
		Name: "resource two",
	}

	actual, err := resources.Create(client.ServiceClient(), createOpts).Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, SecondResource, *actual)
}

func TestDeleteResource(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleDeleteResourceSuccessfully(t)

	res := resources.Delete(client.ServiceClient(), "9fe1d3")
	th.AssertNoErr(t, res.Err)
}

func TestUpdateResource(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleUpdateResourceSuccessfully(t)

	updateOpts := resources.UpdateOpts{
		Description: "Staging Resource",
	}

	actual, err := resources.Update(client.ServiceClient(), "9fe1d3", updateOpts).Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, SecondResourceUpdated, *actual)
}
