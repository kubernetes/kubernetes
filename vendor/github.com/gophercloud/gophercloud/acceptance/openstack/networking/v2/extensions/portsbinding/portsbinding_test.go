// +build acceptance networking

package portsbinding

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	networking "github.com/gophercloud/gophercloud/acceptance/openstack/networking/v2"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/portsbinding"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/ports"
)

func TestPortsbindingCRUD(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	if err != nil {
		t.Fatalf("Unable to create a network client: %v", err)
	}

	// Create Network
	network, err := networking.CreateNetwork(t, client)
	if err != nil {
		t.Fatalf("Unable to create network: %v", err)
	}
	defer networking.DeleteNetwork(t, client, network.ID)

	// Create Subnet
	subnet, err := networking.CreateSubnet(t, client, network.ID)
	if err != nil {
		t.Fatalf("Unable to create subnet: %v", err)
	}
	defer networking.DeleteSubnet(t, client, subnet.ID)

	// Define a host
	hostID := "localhost"

	// Create port
	port, err := CreatePortsbinding(t, client, network.ID, subnet.ID, hostID)
	if err != nil {
		t.Fatalf("Unable to create port: %v", err)
	}
	defer networking.DeletePort(t, client, port.ID)

	PrintPortsbinding(t, port)

	// Update port
	newPortName := tools.RandomString("TESTACC-", 8)
	updateOpts := ports.UpdateOpts{
		Name: newPortName,
	}
	newPort, err := portsbinding.Update(client, port.ID, updateOpts).Extract()
	if err != nil {
		t.Fatalf("Could not update port: %v", err)
	}

	PrintPortsbinding(t, newPort)
}
