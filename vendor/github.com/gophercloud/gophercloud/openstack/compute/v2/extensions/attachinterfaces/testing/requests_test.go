package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/attachinterfaces"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

func TestListInterface(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleInterfaceListSuccessfully(t)

	expected := ListInterfacesExpected
	pages := 0
	err := attachinterfaces.List(client.ServiceClient(), "b07e7a3b-d951-4efc-a4f9-ac9f001afb7f").EachPage(func(page pagination.Page) (bool, error) {
		pages++

		actual, err := attachinterfaces.ExtractInterfaces(page)
		th.AssertNoErr(t, err)

		if len(actual) != 1 {
			t.Fatalf("Expected 1 interface, got %d", len(actual))
		}
		th.CheckDeepEquals(t, expected, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.CheckEquals(t, 1, pages)
}

func TestListInterfacesAllPages(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleInterfaceListSuccessfully(t)

	allPages, err := attachinterfaces.List(client.ServiceClient(), "b07e7a3b-d951-4efc-a4f9-ac9f001afb7f").AllPages()
	th.AssertNoErr(t, err)
	_, err = attachinterfaces.ExtractInterfaces(allPages)
	th.AssertNoErr(t, err)
}
