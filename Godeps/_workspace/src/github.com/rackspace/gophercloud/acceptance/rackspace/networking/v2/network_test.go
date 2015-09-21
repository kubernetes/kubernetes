// +build acceptance networking

package v2

import (
	"strconv"
	"testing"

	os "github.com/rackspace/gophercloud/openstack/networking/v2/networks"
	"github.com/rackspace/gophercloud/pagination"
	"github.com/rackspace/gophercloud/rackspace/networking/v2/networks"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestNetworkCRUDOperations(t *testing.T) {
	Setup(t)
	defer Teardown()

	// Create a network
	n, err := networks.Create(Client, os.CreateOpts{Name: "sample_network", AdminStateUp: os.Up}).Extract()
	th.AssertNoErr(t, err)
	defer networks.Delete(Client, n.ID)
	th.AssertEquals(t, "sample_network", n.Name)
	th.AssertEquals(t, true, n.AdminStateUp)
	networkID := n.ID

	// List networks
	pager := networks.List(Client, os.ListOpts{Limit: 2})
	err = pager.EachPage(func(page pagination.Page) (bool, error) {
		t.Logf("--- Page ---")

		networkList, err := os.ExtractNetworks(page)
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
	th.AssertEquals(t, "ACTIVE", n.Status)
	th.AssertDeepEquals(t, []string{}, n.Subnets)
	th.AssertEquals(t, "sample_network", n.Name)
	th.AssertEquals(t, true, n.AdminStateUp)
	th.AssertEquals(t, false, n.Shared)
	th.AssertEquals(t, networkID, n.ID)

	// Update network
	n, err = networks.Update(Client, networkID, os.UpdateOpts{Name: "new_network_name"}).Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, "new_network_name", n.Name)

	// Delete network
	res := networks.Delete(Client, networkID)
	th.AssertNoErr(t, res.Err)
}
