// +build acceptance compute servers

package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/floatingips"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/servers"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestFloatingIPsCreateDelete(t *testing.T) {
	client, err := clients.NewComputeV2Client()
	th.AssertNoErr(t, err)

	floatingIP, err := CreateFloatingIP(t, client)
	th.AssertNoErr(t, err)
	defer DeleteFloatingIP(t, client, floatingIP)

	tools.PrintResource(t, floatingIP)

	allPages, err := floatingips.List(client).AllPages()
	th.AssertNoErr(t, err)

	allFloatingIPs, err := floatingips.ExtractFloatingIPs(allPages)
	th.AssertNoErr(t, err)

	var found bool
	for _, fip := range allFloatingIPs {
		tools.PrintResource(t, floatingIP)

		if fip.ID == floatingIP.ID {
			found = true
		}
	}

	th.AssertEquals(t, found, true)

	fip, err := floatingips.Get(client, floatingIP.ID).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, floatingIP.ID, fip.ID)
}

func TestFloatingIPsAssociate(t *testing.T) {
	clients.RequireLong(t)

	client, err := clients.NewComputeV2Client()
	th.AssertNoErr(t, err)

	server, err := CreateServer(t, client)
	th.AssertNoErr(t, err)
	defer DeleteServer(t, client, server)

	floatingIP, err := CreateFloatingIP(t, client)
	th.AssertNoErr(t, err)
	defer DeleteFloatingIP(t, client, floatingIP)

	tools.PrintResource(t, floatingIP)

	err = AssociateFloatingIP(t, client, floatingIP, server)
	th.AssertNoErr(t, err)
	defer DisassociateFloatingIP(t, client, floatingIP, server)

	newFloatingIP, err := floatingips.Get(client, floatingIP.ID).Extract()
	th.AssertNoErr(t, err)

	t.Logf("Floating IP %s is associated with Fixed IP %s", floatingIP.IP, newFloatingIP.FixedIP)

	tools.PrintResource(t, newFloatingIP)

	th.AssertEquals(t, newFloatingIP.InstanceID, server.ID)
}

func TestFloatingIPsFixedIPAssociate(t *testing.T) {
	clients.RequireLong(t)

	client, err := clients.NewComputeV2Client()
	th.AssertNoErr(t, err)

	choices, err := clients.AcceptanceTestChoicesFromEnv()
	th.AssertNoErr(t, err)

	server, err := CreateServer(t, client)
	th.AssertNoErr(t, err)
	defer DeleteServer(t, client, server)

	newServer, err := servers.Get(client, server.ID).Extract()
	th.AssertNoErr(t, err)

	floatingIP, err := CreateFloatingIP(t, client)
	th.AssertNoErr(t, err)
	defer DeleteFloatingIP(t, client, floatingIP)

	tools.PrintResource(t, floatingIP)

	var fixedIP string
	for _, networkAddresses := range newServer.Addresses[choices.NetworkName].([]interface{}) {
		address := networkAddresses.(map[string]interface{})
		if address["OS-EXT-IPS:type"] == "fixed" {
			if address["version"].(float64) == 4 {
				fixedIP = address["addr"].(string)
			}
		}
	}

	err = AssociateFloatingIPWithFixedIP(t, client, floatingIP, newServer, fixedIP)
	th.AssertNoErr(t, err)
	defer DisassociateFloatingIP(t, client, floatingIP, newServer)

	newFloatingIP, err := floatingips.Get(client, floatingIP.ID).Extract()
	th.AssertNoErr(t, err)

	t.Logf("Floating IP %s is associated with Fixed IP %s", floatingIP.IP, newFloatingIP.FixedIP)

	tools.PrintResource(t, newFloatingIP)

	th.AssertEquals(t, newFloatingIP.InstanceID, server.ID)
	th.AssertEquals(t, newFloatingIP.FixedIP, fixedIP)
}
