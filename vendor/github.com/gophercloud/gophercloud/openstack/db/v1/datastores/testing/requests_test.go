package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud/openstack/db/v1/datastores"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
	"github.com/gophercloud/gophercloud/testhelper/fixture"
)

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	fixture.SetupHandler(t, "/datastores", "GET", "", ListDSResp, 200)

	pages := 0

	err := datastores.List(fake.ServiceClient()).EachPage(func(page pagination.Page) (bool, error) {
		pages++

		actual, err := datastores.ExtractDatastores(page)
		if err != nil {
			return false, err
		}

		th.CheckDeepEquals(t, []datastores.Datastore{ExampleDatastore}, actual)

		return true, nil
	})

	th.AssertNoErr(t, err)
	th.AssertEquals(t, 1, pages)
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	fixture.SetupHandler(t, "/datastores/{dsID}", "GET", "", GetDSResp, 200)

	ds, err := datastores.Get(fake.ServiceClient(), "{dsID}").Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, &ExampleDatastore, ds)
}

func TestListVersions(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	fixture.SetupHandler(t, "/datastores/{dsID}/versions", "GET", "", ListVersionsResp, 200)

	pages := 0

	err := datastores.ListVersions(fake.ServiceClient(), "{dsID}").EachPage(func(page pagination.Page) (bool, error) {
		pages++

		actual, err := datastores.ExtractVersions(page)
		if err != nil {
			return false, err
		}

		th.CheckDeepEquals(t, ExampleVersions, actual)

		return true, nil
	})

	th.AssertNoErr(t, err)
	th.AssertEquals(t, 1, pages)
}

func TestGetVersion(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	fixture.SetupHandler(t, "/datastores/{dsID}/versions/{versionID}", "GET", "", GetVersionResp, 200)

	ds, err := datastores.GetVersion(fake.ServiceClient(), "{dsID}", "{versionID}").Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, &ExampleVersion1, ds)
}
