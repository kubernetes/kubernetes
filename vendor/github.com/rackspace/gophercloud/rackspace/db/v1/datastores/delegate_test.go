package datastores

import (
	"testing"

	os "github.com/rackspace/gophercloud/openstack/db/v1/datastores"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
	fake "github.com/rackspace/gophercloud/testhelper/client"
	"github.com/rackspace/gophercloud/testhelper/fixture"
)

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	fixture.SetupHandler(t, "/datastores", "GET", "", os.ListDSResp, 200)

	pages := 0

	err := List(fake.ServiceClient()).EachPage(func(page pagination.Page) (bool, error) {
		pages++

		actual, err := os.ExtractDatastores(page)
		if err != nil {
			return false, err
		}

		th.CheckDeepEquals(t, []os.Datastore{os.ExampleDatastore}, actual)

		return true, nil
	})

	th.AssertNoErr(t, err)
	th.AssertEquals(t, 1, pages)
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	fixture.SetupHandler(t, "/datastores/{dsID}", "GET", "", os.GetDSResp, 200)

	ds, err := Get(fake.ServiceClient(), "{dsID}").Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, &os.ExampleDatastore, ds)
}

func TestListVersions(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	fixture.SetupHandler(t, "/datastores/{dsID}/versions", "GET", "", os.ListVersionsResp, 200)

	pages := 0

	err := ListVersions(fake.ServiceClient(), "{dsID}").EachPage(func(page pagination.Page) (bool, error) {
		pages++

		actual, err := os.ExtractVersions(page)
		if err != nil {
			return false, err
		}

		th.CheckDeepEquals(t, os.ExampleVersions, actual)

		return true, nil
	})

	th.AssertNoErr(t, err)
	th.AssertEquals(t, 1, pages)
}

func TestGetVersion(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	fixture.SetupHandler(t, "/datastores/{dsID}/versions/{versionID}", "GET", "", os.GetVersionResp, 200)

	ds, err := GetVersion(fake.ServiceClient(), "{dsID}", "{versionID}").Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, &os.ExampleVersion1, ds)
}
