// +build acceptance networking layer3 floatingips

package layer3

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	networking "github.com/gophercloud/gophercloud/acceptance/openstack/networking/v2"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/layer3/floatingips"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/subnets"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestLayer3FloatingIPsCreateDelete(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	th.AssertNoErr(t, err)

	choices, err := clients.AcceptanceTestChoicesFromEnv()
	th.AssertNoErr(t, err)

	fip, err := CreateFloatingIP(t, client, choices.ExternalNetworkID, "")
	th.AssertNoErr(t, err)
	defer DeleteFloatingIP(t, client, fip.ID)

	newFip, err := floatingips.Get(client, fip.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newFip)

	allPages, err := floatingips.List(client, floatingips.ListOpts{}).AllPages()
	th.AssertNoErr(t, err)

	allFIPs, err := floatingips.ExtractFloatingIPs(allPages)
	th.AssertNoErr(t, err)

	var found bool
	for _, fip := range allFIPs {
		if fip.ID == newFip.ID {
			found = true
		}
	}

	th.AssertEquals(t, found, true)
}

func TestLayer3FloatingIPsExternalCreateDelete(t *testing.T) {
	clients.RequireAdmin(t)

	client, err := clients.NewNetworkV2Client()
	th.AssertNoErr(t, err)

	choices, err := clients.AcceptanceTestChoicesFromEnv()
	th.AssertNoErr(t, err)

	// Create Network
	network, err := networking.CreateNetwork(t, client)
	th.AssertNoErr(t, err)
	defer networking.DeleteNetwork(t, client, network.ID)

	subnet, err := networking.CreateSubnet(t, client, network.ID)
	th.AssertNoErr(t, err)
	defer networking.DeleteSubnet(t, client, subnet.ID)

	router, err := CreateExternalRouter(t, client)
	th.AssertNoErr(t, err)
	defer DeleteRouter(t, client, router.ID)

	port, err := networking.CreatePort(t, client, network.ID, subnet.ID)
	th.AssertNoErr(t, err)
	// not required, since "DeleteRouterInterface" above removes the port
	// defer networking.DeletePort(t, client, port.ID)

	_, err = CreateRouterInterface(t, client, port.ID, router.ID)
	th.AssertNoErr(t, err)
	defer DeleteRouterInterface(t, client, port.ID, router.ID)

	fip, err := CreateFloatingIP(t, client, choices.ExternalNetworkID, port.ID)
	th.AssertNoErr(t, err)
	defer DeleteFloatingIP(t, client, fip.ID)

	newFip, err := floatingips.Get(client, fip.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newFip)

	// Disassociate the floating IP
	updateOpts := floatingips.UpdateOpts{
		PortID: new(string),
	}

	_, err = floatingips.Update(client, fip.ID, updateOpts).Extract()
	th.AssertNoErr(t, err)

	newFip, err = floatingips.Get(client, fip.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newFip)

	th.AssertEquals(t, newFip.PortID, "")
}

func TestLayer3FloatingIPsWithFixedIPsExternalCreateDelete(t *testing.T) {
	clients.RequireAdmin(t)

	client, err := clients.NewNetworkV2Client()
	th.AssertNoErr(t, err)

	choices, err := clients.AcceptanceTestChoicesFromEnv()
	th.AssertNoErr(t, err)

	// Create Network
	network, err := networking.CreateNetwork(t, client)
	th.AssertNoErr(t, err)
	defer networking.DeleteNetwork(t, client, network.ID)

	subnet, err := networking.CreateSubnet(t, client, network.ID)
	th.AssertNoErr(t, err)
	defer networking.DeleteSubnet(t, client, subnet.ID)

	router, err := CreateExternalRouter(t, client)
	th.AssertNoErr(t, err)
	defer DeleteRouter(t, client, router.ID)

	port, err := networking.CreatePortWithMultipleFixedIPs(t, client, network.ID, subnet.ID)
	th.AssertNoErr(t, err)
	defer networking.DeletePort(t, client, port.ID)

	var fixedIPs []string
	for _, fixedIP := range port.FixedIPs {
		fixedIPs = append(fixedIPs, fixedIP.IPAddress)
	}

	iface, err := CreateRouterInterfaceOnSubnet(t, client, subnet.ID, router.ID)
	th.AssertNoErr(t, err)
	defer DeleteRouterInterface(t, client, iface.PortID, router.ID)

	fip, err := CreateFloatingIPWithFixedIP(t, client, choices.ExternalNetworkID, port.ID, fixedIPs[0])
	th.AssertNoErr(t, err)
	defer DeleteFloatingIP(t, client, fip.ID)

	newFip, err := floatingips.Get(client, fip.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newFip)

	// Associate the floating IP with another fixed IP
	updateOpts := floatingips.UpdateOpts{
		PortID:  &port.ID,
		FixedIP: fixedIPs[1],
	}

	_, err = floatingips.Update(client, fip.ID, updateOpts).Extract()
	th.AssertNoErr(t, err)

	newFip, err = floatingips.Get(client, fip.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newFip)

	th.AssertEquals(t, newFip.FixedIP, fixedIPs[1])

	// Disassociate the floating IP
	updateOpts = floatingips.UpdateOpts{
		PortID: new(string),
	}

	_, err = floatingips.Update(client, fip.ID, updateOpts).Extract()
	th.AssertNoErr(t, err)
}

func TestLayer3FloatingIPsCreateDeleteBySubnetID(t *testing.T) {
	clients.RequireAdmin(t)

	client, err := clients.NewNetworkV2Client()
	th.AssertNoErr(t, err)

	choices, err := clients.AcceptanceTestChoicesFromEnv()
	th.AssertNoErr(t, err)

	listOpts := subnets.ListOpts{
		NetworkID: choices.ExternalNetworkID,
		IPVersion: 4,
	}

	subnetPages, err := subnets.List(client, listOpts).AllPages()
	th.AssertNoErr(t, err)

	allSubnets, err := subnets.ExtractSubnets(subnetPages)
	th.AssertNoErr(t, err)

	createOpts := floatingips.CreateOpts{
		FloatingNetworkID: choices.ExternalNetworkID,
		SubnetID:          allSubnets[0].ID,
	}

	fip, err := floatingips.Create(client, createOpts).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, fip)

	DeleteFloatingIP(t, client, fip.ID)
}
