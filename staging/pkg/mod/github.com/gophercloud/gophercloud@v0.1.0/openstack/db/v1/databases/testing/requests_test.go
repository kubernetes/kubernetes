package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud/openstack/db/v1/databases"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleCreate(t)

	opts := databases.BatchCreateOpts{
		databases.CreateOpts{Name: "testingdb", CharSet: "utf8", Collate: "utf8_general_ci"},
		databases.CreateOpts{Name: "sampledb"},
	}

	res := databases.Create(fake.ServiceClient(), instanceID, opts)
	th.AssertNoErr(t, res.Err)
}

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleList(t)

	expectedDBs := []databases.Database{
		{Name: "anotherexampledb"},
		{Name: "exampledb"},
		{Name: "nextround"},
		{Name: "sampledb"},
		{Name: "testingdb"},
	}

	pages := 0
	err := databases.List(fake.ServiceClient(), instanceID).EachPage(func(page pagination.Page) (bool, error) {
		pages++

		actual, err := databases.ExtractDBs(page)
		if err != nil {
			return false, err
		}

		th.CheckDeepEquals(t, expectedDBs, actual)

		return true, nil
	})

	th.AssertNoErr(t, err)

	if pages != 1 {
		t.Errorf("Expected 1 page, saw %d", pages)
	}
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleDelete(t)

	err := databases.Delete(fake.ServiceClient(), instanceID, "{dbName}").ExtractErr()
	th.AssertNoErr(t, err)
}
