// +build acceptance networking layer3 floatingips

package layer3

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	networking "github.com/gophercloud/gophercloud/acceptance/openstack/networking/v2"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/layer3/floatingips"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/networks"
)

func TestLayer3FloatingIPsList(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	listOpts := floatingips.ListOpts{
		Status: "DOWN",
	}
	allPages, err := floatingips.List(client, listOpts).AllPages()
	if err != nil {
		t.Fatalf("Unable to list floating IPs: %v", err)
	}

	allFIPs, err := floatingips.ExtractFloatingIPs(allPages)
	if err != nil {
		t.Fatalf("Unable to extract floating IPs: %v", err)
	}

	for _, fip := range allFIPs {
		tools.PrintResource(t, fip)
	}
}

func TestLayer3FloatingIPsCreateDelete(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	choices, err := clients.AcceptanceTestChoicesFromEnv()
	if err != nil {
		t.Fatalf("Unable to get choices: %v", err)
	}

	netid, err := networks.IDFromName(client, choices.NetworkName)
	if err != nil {
		t.Fatalf("Unable to find network id: %v", err)
	}

	subnet, err := networking.CreateSubnet(t, client, netid)
	if err != nil {
		t.Fatalf("Unable to create subnet: %v", err)
	}
	defer networking.DeleteSubnet(t, client, subnet.ID)

	router, err := CreateExternalRouter(t, client)
	if err != nil {
		t.Fatalf("Unable to create router: %v", err)
	}
	defer DeleteRouter(t, client, router.ID)

	port, err := networking.CreatePort(t, client, netid, subnet.ID)
	if err != nil {
		t.Fatalf("Unable to create port: %v", err)
	}

	_, err = CreateRouterInterface(t, client, port.ID, router.ID)
	if err != nil {
		t.Fatalf("Unable to create router interface: %v", err)
	}
	defer DeleteRouterInterface(t, client, port.ID, router.ID)

	fip, err := CreateFloatingIP(t, client, choices.ExternalNetworkID, port.ID)
	if err != nil {
		t.Fatalf("Unable to create floating IP: %v", err)
	}
	defer DeleteFloatingIP(t, client, fip.ID)

	newFip, err := floatingips.Get(client, fip.ID).Extract()
	if err != nil {
		t.Fatalf("Unable to get floating ip: %v", err)
	}

	tools.PrintResource(t, newFip)

	// Disassociate the floating IP
	updateOpts := floatingips.UpdateOpts{
		PortID: nil,
	}

	newFip, err = floatingips.Update(client, fip.ID, updateOpts).Extract()
	if err != nil {
		t.Fatalf("Unable to disassociate floating IP: %v", err)
	}
}
