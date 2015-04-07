// +build acceptance

package v1

import (
	"testing"

	"github.com/rackspace/gophercloud"
	os "github.com/rackspace/gophercloud/openstack/cdn/v1/flavors"
	"github.com/rackspace/gophercloud/pagination"
	"github.com/rackspace/gophercloud/rackspace/cdn/v1/flavors"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestFlavor(t *testing.T) {
	client := newClient(t)

	t.Log("Listing Flavors")
	id := testFlavorsList(t, client)

	t.Log("Retrieving Flavor")
	testFlavorGet(t, client, id)
}

func testFlavorsList(t *testing.T, client *gophercloud.ServiceClient) string {
	var id string
	err := flavors.List(client).EachPage(func(page pagination.Page) (bool, error) {
		flavorList, err := os.ExtractFlavors(page)
		th.AssertNoErr(t, err)

		for _, flavor := range flavorList {
			t.Logf("Listing flavor: ID [%s] Providers [%+v]", flavor.ID, flavor.Providers)
			id = flavor.ID
		}

		return true, nil
	})

	th.AssertNoErr(t, err)
	return id
}

func testFlavorGet(t *testing.T, client *gophercloud.ServiceClient, id string) {
	flavor, err := flavors.Get(client, id).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Retrieved Flavor: %+v", *flavor)
}
