// +build acceptance rackspace

package v2

import (
	"testing"

	"github.com/rackspace/gophercloud/pagination"
	"github.com/rackspace/gophercloud/rackspace/compute/v2/networks"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestNetworks(t *testing.T) {
	client, err := newClient()
	th.AssertNoErr(t, err)

	// Create a network
	n, err := networks.Create(client, networks.CreateOpts{Label: "sample_network", CIDR: "172.20.0.0/24"}).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Created network: %+v\n", n)
	defer networks.Delete(client, n.ID)
	th.AssertEquals(t, n.Label, "sample_network")
	th.AssertEquals(t, n.CIDR, "172.20.0.0/24")
	networkID := n.ID

	// List networks
	pager := networks.List(client)
	err = pager.EachPage(func(page pagination.Page) (bool, error) {
		t.Logf("--- Page ---")

		networkList, err := networks.ExtractNetworks(page)
		th.AssertNoErr(t, err)

		for _, n := range networkList {
			t.Logf("Network: ID [%s] Label [%s] CIDR [%s]",
				n.ID, n.Label, n.CIDR)
		}

		return true, nil
	})
	th.CheckNoErr(t, err)

	// Get a network
	if networkID == "" {
		t.Fatalf("In order to retrieve a network, the NetworkID must be set")
	}
	n, err = networks.Get(client, networkID).Extract()
	t.Logf("Retrieved Network: %+v\n", n)
	th.AssertNoErr(t, err)
	th.AssertEquals(t, n.CIDR, "172.20.0.0/24")
	th.AssertEquals(t, n.Label, "sample_network")
	th.AssertEquals(t, n.ID, networkID)
}
