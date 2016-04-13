// +build acceptance db rackspace

package v1

import (
	os "github.com/rackspace/gophercloud/openstack/db/v1/flavors"
	"github.com/rackspace/gophercloud/pagination"
	"github.com/rackspace/gophercloud/rackspace/db/v1/flavors"
)

func (c context) listFlavors() {
	c.Logf("Listing flavors")

	err := flavors.List(c.client).EachPage(func(page pagination.Page) (bool, error) {
		flavorList, err := os.ExtractFlavors(page)
		c.AssertNoErr(err)

		for _, f := range flavorList {
			c.Logf("Flavor: ID [%s] Name [%s] RAM [%d]", f.ID, f.Name, f.RAM)
		}

		return true, nil
	})

	c.AssertNoErr(err)
}

func (c context) getFlavor() {
	flavor, err := flavors.Get(c.client, "1").Extract()
	c.Logf("Getting flavor %s", flavor.ID)
	c.AssertNoErr(err)
}
