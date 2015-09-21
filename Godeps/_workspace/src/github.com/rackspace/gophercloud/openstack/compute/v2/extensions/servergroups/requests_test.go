package servergroups

import (
	"testing"

	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
	"github.com/rackspace/gophercloud/testhelper/client"
)

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListSuccessfully(t)

	count := 0
	err := List(client.ServiceClient()).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := ExtractServerGroups(page)
		th.AssertNoErr(t, err)
		th.CheckDeepEquals(t, ExpectedServerGroupSlice, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.CheckEquals(t, 1, count)
}

func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleCreateSuccessfully(t)

	actual, err := Create(client.ServiceClient(), CreateOpts{
		Name:     "test",
		Policies: []string{"anti-affinity"},
	}).Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, &CreatedServerGroup, actual)
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleGetSuccessfully(t)

	actual, err := Get(client.ServiceClient(), "4d8c3732-a248-40ed-bebc-539a6ffd25c0").Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, &FirstServerGroup, actual)
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleDeleteSuccessfully(t)

	err := Delete(client.ServiceClient(), "616fb98f-46ca-475e-917e-2563e5a8cd19").ExtractErr()
	th.AssertNoErr(t, err)
}
