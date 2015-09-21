// +build acceptance networking

package v2

import (
	"strconv"
	"testing"

	"github.com/rackspace/gophercloud/openstack/networking/v2/networks"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestNetworkCRUDOperations(t *testing.T) {
	Setup(t)
	defer Teardown()

	// Create a network
	n, err := networks.Create(Client, networks.CreateOpts{Name: "sample_network", AdminStateUp: networks.Up}).Extract()
	th.AssertNoErr(t, err)
	defer networks.Delete(Client, n.ID)
	th.AssertEquals(t, n.Name, "sample_network")
	th.AssertEquals(t, n.AdminStateUp, true)
	networkID := n.ID

	// List networks
	pager := networks.List(Client, networks.ListOpts{Limit: 2})
	err = pager.EachPage(func(page pagination.Page) (bool, error) {
		t.Logf("--- Page ---")

		networkList, err := networks.ExtractNetworks(page)
		th.AssertNoErr(t, err)

		for _, n := range networkList {
			t.Logf("Network: ID [%s] Name [%s] Status [%s] Is shared? [%s]",
				n.ID, n.Name, n.Status, strconv.FormatBool(n.Shared))
		}

		return true, nil
	})
	th.CheckNoErr(t, err)

	// Get a network
	if networkID == "" {
		t.Fatalf("In order to retrieve a network, the NetworkID must be set")
	}
	n, err = networks.Get(Client, networkID).Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, n.Status, "ACTIVE")
	th.AssertDeepEquals(t, n.Subnets, []string{})
	th.AssertEquals(t, n.Name, "sample_network")
	th.AssertEquals(t, n.AdminStateUp, true)
	th.AssertEquals(t, n.Shared, false)
	th.AssertEquals(t, n.ID, networkID)

	// Update network
	n, err = networks.Update(Client, networkID, networks.UpdateOpts{Name: "new_network_name"}).Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, n.Name, "new_network_name")

	// Delete network
	res := networks.Delete(Client, networkID)
	th.AssertNoErr(t, res.Err)
}

func TestCreateMultipleNetworks(t *testing.T) {
	//networks.CreateMany()
}
