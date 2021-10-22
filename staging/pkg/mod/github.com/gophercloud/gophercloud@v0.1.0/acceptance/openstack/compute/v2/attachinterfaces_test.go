// +build acceptance compute servers

package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/servers"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestAttachDetachInterface(t *testing.T) {
	clients.RequireLong(t)

	choices, err := clients.AcceptanceTestChoicesFromEnv()
	th.AssertNoErr(t, err)

	client, err := clients.NewComputeV2Client()
	th.AssertNoErr(t, err)

	server, err := CreateServer(t, client)
	th.AssertNoErr(t, err)
	defer DeleteServer(t, client, server)

	iface, err := AttachInterface(t, client, server.ID)
	th.AssertNoErr(t, err)
	defer DetachInterface(t, client, server.ID, iface.PortID)

	tools.PrintResource(t, iface)

	server, err = servers.Get(client, server.ID).Extract()
	th.AssertNoErr(t, err)

	var found bool
	for _, networkAddresses := range server.Addresses[choices.NetworkName].([]interface{}) {
		address := networkAddresses.(map[string]interface{})
		if address["OS-EXT-IPS:type"] == "fixed" {
			fixedIP := address["addr"].(string)

			for _, v := range iface.FixedIPs {
				if fixedIP == v.IPAddress {
					found = true
				}
			}
		}
	}

	th.AssertEquals(t, found, true)
}
