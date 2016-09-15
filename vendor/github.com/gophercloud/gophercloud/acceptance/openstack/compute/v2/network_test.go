// +build acceptance compute servers

package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/networks"
)

func TestNetworksList(t *testing.T) {
	client, err := clients.NewComputeV2Client()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	allPages, err := networks.List(client).AllPages()
	if err != nil {
		t.Fatalf("Unable to list networks: %v", err)
	}

	allNetworks, err := networks.ExtractNetworks(allPages)
	if err != nil {
		t.Fatalf("Unable to list networks: %v", err)
	}

	for _, network := range allNetworks {
		PrintNetwork(t, &network)
	}
}

func TestNetworksGet(t *testing.T) {
	choices, err := clients.AcceptanceTestChoicesFromEnv()
	if err != nil {
		t.Fatal(err)
	}

	client, err := clients.NewComputeV2Client()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	networkID, err := GetNetworkIDFromNetworks(t, client, choices.NetworkName)
	if err != nil {
		t.Fatal(err)
	}

	network, err := networks.Get(client, networkID).Extract()
	if err != nil {
		t.Fatalf("Unable to get network %s: %v", networkID, err)
	}

	PrintNetwork(t, network)
}
