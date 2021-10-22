// +build acceptance networking vlantransparent

package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	networkingv2 "github.com/gophercloud/gophercloud/acceptance/openstack/networking/v2"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestVLANTransparentCRUD(t *testing.T) {
	t.Skip("We don't have VLAN transparent extension in OpenLab.")

	client, err := clients.NewNetworkV2Client()
	th.AssertNoErr(t, err)

	// Create a VLAN transparent network.
	network, err := CreateVLANTransparentNetwork(t, client)
	th.AssertNoErr(t, err)
	defer networkingv2.DeleteNetwork(t, client, network.ID)

	tools.PrintResource(t, network)

	// Update the created VLAN transparent network.
	newNetwork, err := UpdateVLANTransparentNetwork(t, client, network.ID)
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newNetwork)

	// Check that the created VLAN transparent network exists.
	vlanTransparentNetworks, err := ListVLANTransparentNetworks(t, client)
	th.AssertNoErr(t, err)

	var found bool
	for _, vlanTransparentNetwork := range vlanTransparentNetworks {
		if vlanTransparentNetwork.ID == network.ID {
			found = true
		}
	}

	th.AssertEquals(t, found, true)
}
