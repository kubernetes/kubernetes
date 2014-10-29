// +build acceptance rackspace

package v2

import (
	"testing"

	"github.com/rackspace/gophercloud/pagination"
	"github.com/rackspace/gophercloud/rackspace/compute/v2/networks"
	"github.com/rackspace/gophercloud/rackspace/compute/v2/virtualinterfaces"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestVirtualInterfaces(t *testing.T) {
	client, err := newClient()
	th.AssertNoErr(t, err)

	// Create a server
	server := createServer(t, client, "")
	t.Logf("Created Server: %v\n", server)
	defer deleteServer(t, client, server)
	serverID := server.ID

	// Create a network
	n, err := networks.Create(client, networks.CreateOpts{Label: "sample_network", CIDR: "172.20.0.0/24"}).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Created Network: %v\n", n)
	defer networks.Delete(client, n.ID)
	networkID := n.ID

	// Create a virtual interface
	vi, err := virtualinterfaces.Create(client, serverID, networkID).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Created virtual interface: %+v\n", vi)
	defer virtualinterfaces.Delete(client, serverID, vi.ID)

	// List virtual interfaces
	pager := virtualinterfaces.List(client, serverID)
	err = pager.EachPage(func(page pagination.Page) (bool, error) {
		t.Logf("--- Page ---")

		virtualinterfacesList, err := virtualinterfaces.ExtractVirtualInterfaces(page)
		th.AssertNoErr(t, err)

		for _, vi := range virtualinterfacesList {
			t.Logf("Virtual Interface: ID [%s] MAC Address [%s] IP Addresses [%v]",
				vi.ID, vi.MACAddress, vi.IPAddresses)
		}

		return true, nil
	})
	th.CheckNoErr(t, err)
}
