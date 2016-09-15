// +build acceptance networking

package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/ports"
)

func TestPortsList(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	if err != nil {
		t.Fatalf("Unable to create a network client: %v", err)
	}

	allPages, err := ports.List(client, nil).AllPages()
	if err != nil {
		t.Fatalf("Unable to list ports: %v", err)
	}

	allPorts, err := ports.ExtractPorts(allPages)
	if err != nil {
		t.Fatalf("Unable to extract ports: %v", err)
	}

	for _, port := range allPorts {
		PrintPort(t, &port)
	}
}

func TestPortsCRUD(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	if err != nil {
		t.Fatalf("Unable to create a network client: %v", err)
	}

	// Create Network
	network, err := CreateNetwork(t, client)
	if err != nil {
		t.Fatalf("Unable to create network: %v", err)
	}
	defer DeleteNetwork(t, client, network.ID)

	// Create Subnet
	subnet, err := CreateSubnet(t, client, network.ID)
	if err != nil {
		t.Fatalf("Unable to create subnet: %v", err)
	}
	defer DeleteSubnet(t, client, subnet.ID)

	// Create port
	port, err := CreatePort(t, client, network.ID, subnet.ID)
	if err != nil {
		t.Fatalf("Unable to create port: %v", err)
	}
	defer DeletePort(t, client, port.ID)

	PrintPort(t, port)

	// Update port
	newPortName := tools.RandomString("TESTACC-", 8)
	updateOpts := ports.UpdateOpts{
		Name: newPortName,
	}
	newPort, err := ports.Update(client, port.ID, updateOpts).Extract()
	if err != nil {
		t.Fatalf("Could not update port: %v", err)
	}

	PrintPort(t, newPort)
}
