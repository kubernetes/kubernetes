// +build acceptance compute servers

package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/networks"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestNetworksList(t *testing.T) {
	client, err := clients.NewComputeV2Client()
	th.AssertNoErr(t, err)

	choices, err := clients.AcceptanceTestChoicesFromEnv()
	th.AssertNoErr(t, err)

	allPages, err := networks.List(client).AllPages()
	th.AssertNoErr(t, err)

	allNetworks, err := networks.ExtractNetworks(allPages)
	th.AssertNoErr(t, err)

	var found bool
	for _, network := range allNetworks {
		tools.PrintResource(t, network)

		if network.Label == choices.NetworkName {
			found = true
		}
	}

	th.AssertEquals(t, found, true)
}

func TestNetworksGet(t *testing.T) {
	choices, err := clients.AcceptanceTestChoicesFromEnv()
	th.AssertNoErr(t, err)

	client, err := clients.NewComputeV2Client()
	th.AssertNoErr(t, err)

	networkID, err := GetNetworkIDFromNetworks(t, client, choices.NetworkName)
	th.AssertNoErr(t, err)

	network, err := networks.Get(client, networkID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, network)

	th.AssertEquals(t, network.Label, choices.NetworkName)
}
