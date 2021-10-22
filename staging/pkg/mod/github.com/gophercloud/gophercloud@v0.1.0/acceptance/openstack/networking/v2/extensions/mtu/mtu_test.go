// +build acceptance networking

package mtu

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	networking "github.com/gophercloud/gophercloud/acceptance/openstack/networking/v2"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/common/extensions"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/mtu"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/networks"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestMTUNetworkCRUDL(t *testing.T) {
	clients.RequireAdmin(t)

	client, err := clients.NewNetworkV2Client()
	th.AssertNoErr(t, err)

	extension, err := extensions.Get(client, "net-mtu").Extract()
	if err != nil {
		t.Skip("This test requires net-mtu Neutron extension")
	}
	tools.PrintResource(t, extension)

	mtuWritable, _ := extensions.Get(client, "net-mtu-writable").Extract()
	tools.PrintResource(t, mtuWritable)

	// Create Network
	var networkMTU int
	if mtuWritable != nil {
		networkMTU = 1449
	}
	network, err := CreateNetworkWithMTU(t, client, &networkMTU)
	th.AssertNoErr(t, err)
	defer networking.DeleteNetwork(t, client, network.ID)

	// MTU filtering is supported only in read-only MTU extension
	// https://bugs.launchpad.net/neutron/+bug/1818317
	if mtuWritable == nil {
		// List network successfully
		var listOpts networks.ListOptsBuilder
		listOpts = mtu.ListOptsExt{
			ListOptsBuilder: networks.ListOpts{},
			MTU:             networkMTU,
		}
		var listedNetworks []NetworkMTU
		i := 0
		err = networks.List(client, listOpts).EachPage(func(page pagination.Page) (bool, error) {
			i++
			err := networks.ExtractNetworksInto(page, &listedNetworks)
			if err != nil {
				t.Errorf("Failed to extract networks: %v", err)
				return false, err
			}

			tools.PrintResource(t, listedNetworks)

			th.AssertEquals(t, 1, len(listedNetworks))
			th.CheckDeepEquals(t, *network, listedNetworks[0])

			return true, nil
		})
		th.AssertNoErr(t, err)
		th.AssertEquals(t, 1, i)

		// List network unsuccessfully
		listOpts = mtu.ListOptsExt{
			ListOptsBuilder: networks.ListOpts{},
			MTU:             1,
		}
		i = 0
		err = networks.List(client, listOpts).EachPage(func(page pagination.Page) (bool, error) {
			i++
			err := networks.ExtractNetworksInto(page, &listedNetworks)
			if err != nil {
				t.Errorf("Failed to extract networks: %v", err)
				return false, err
			}

			tools.PrintResource(t, listedNetworks)

			th.AssertEquals(t, 1, len(listedNetworks))
			th.CheckDeepEquals(t, *network, listedNetworks[0])

			return true, nil
		})
		th.AssertNoErr(t, err)
		th.AssertEquals(t, 0, i)
	}

	// Get network
	var getNetwork NetworkMTU
	err = networks.Get(client, network.ID).ExtractInto(&getNetwork)
	th.AssertNoErr(t, err)

	tools.PrintResource(t, getNetwork)
	th.AssertDeepEquals(t, *network, getNetwork)

	if mtuWritable != nil {
		// Update network
		newNetworkDescription := ""
		newNetworkMTU := 1350
		networkUpdateOpts := networks.UpdateOpts{
			Description: &newNetworkDescription,
		}
		var updateOpts networks.UpdateOptsBuilder
		updateOpts = mtu.UpdateOptsExt{
			UpdateOptsBuilder: networkUpdateOpts,
			MTU:               newNetworkMTU,
		}

		var newNetwork NetworkMTU
		err = networks.Update(client, network.ID, updateOpts).ExtractInto(&newNetwork)
		th.AssertNoErr(t, err)

		tools.PrintResource(t, newNetwork)
		th.AssertEquals(t, newNetwork.Description, newNetworkDescription)
		th.AssertEquals(t, newNetwork.MTU, newNetworkMTU)

		// Get updated network
		var getNewNetwork NetworkMTU
		err = networks.Get(client, network.ID).ExtractInto(&getNewNetwork)
		th.AssertNoErr(t, err)

		tools.PrintResource(t, getNewNetwork)
		th.AssertDeepEquals(t, newNetwork, getNewNetwork)
	}
}
