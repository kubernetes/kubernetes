package databases

import (
	"testing"

	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
	fake "github.com/rackspace/gophercloud/testhelper/client"
)

func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleCreate(t)

	opts := BatchCreateOpts{
		CreateOpts{Name: "testingdb", CharSet: "utf8", Collate: "utf8_general_ci"},
		CreateOpts{Name: "sampledb"},
	}

	res := Create(fake.ServiceClient(), instanceID, opts)
	th.AssertNoErr(t, res.Err)
}

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleList(t)

	expectedDBs := []Database{
		Database{Name: "anotherexampledb"},
		Database{Name: "exampledb"},
		Database{Name: "nextround"},
		Database{Name: "sampledb"},
		Database{Name: "testingdb"},
	}

	pages := 0
	err := List(fake.ServiceClient(), instanceID).EachPage(func(page pagination.Page) (bool, error) {
		pages++

		actual, err := ExtractDBs(page)
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

	err := Delete(fake.ServiceClient(), instanceID, "{dbName}").ExtractErr()
	th.AssertNoErr(t, err)
}
