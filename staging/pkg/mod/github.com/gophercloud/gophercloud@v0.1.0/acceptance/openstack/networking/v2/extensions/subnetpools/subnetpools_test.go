// +build acceptance networking subnetpools

package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/subnetpools"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestSubnetPoolsCRUD(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	th.AssertNoErr(t, err)

	// Create a subnetpool
	subnetPool, err := CreateSubnetPool(t, client)
	th.AssertNoErr(t, err)
	defer DeleteSubnetPool(t, client, subnetPool.ID)

	tools.PrintResource(t, subnetPool)

	newName := tools.RandomString("TESTACC-", 8)
	newDescription := ""
	updateOpts := &subnetpools.UpdateOpts{
		Name:        newName,
		Description: &newDescription,
	}

	_, err = subnetpools.Update(client, subnetPool.ID, updateOpts).Extract()
	th.AssertNoErr(t, err)

	newSubnetPool, err := subnetpools.Get(client, subnetPool.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newSubnetPool)
	th.AssertEquals(t, newSubnetPool.Name, newName)
	th.AssertEquals(t, newSubnetPool.Description, newDescription)

	allPages, err := subnetpools.List(client, nil).AllPages()
	th.AssertNoErr(t, err)

	allSubnetPools, err := subnetpools.ExtractSubnetPools(allPages)
	th.AssertNoErr(t, err)

	var found bool
	for _, subnetpool := range allSubnetPools {
		if subnetpool.ID == newSubnetPool.ID {
			found = true
		}
	}

	th.AssertEquals(t, found, true)
}
