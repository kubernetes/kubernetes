package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud/openstack/blockstorage/extensions/schedulerstats"
	"github.com/gophercloud/gophercloud/pagination"
	"github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

func TestListStoragePoolsDetail(t *testing.T) {
	testhelper.SetupHTTP()
	defer testhelper.TeardownHTTP()
	HandleStoragePoolsListSuccessfully(t)

	pages := 0
	err := schedulerstats.List(client.ServiceClient(), schedulerstats.ListOpts{Detail: true}).EachPage(func(page pagination.Page) (bool, error) {
		pages++

		actual, err := schedulerstats.ExtractStoragePools(page)
		testhelper.AssertNoErr(t, err)

		if len(actual) != 2 {
			t.Fatalf("Expected 2 backends, got %d", len(actual))
		}
		testhelper.CheckDeepEquals(t, StoragePoolFake1, actual[0])
		testhelper.CheckDeepEquals(t, StoragePoolFake2, actual[1])

		return true, nil
	})

	testhelper.AssertNoErr(t, err)

	if pages != 1 {
		t.Errorf("Expected 1 page, saw %d", pages)
	}
}
