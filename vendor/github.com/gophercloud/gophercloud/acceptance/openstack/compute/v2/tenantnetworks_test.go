// +build acceptance compute servers

package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/tenantnetworks"
)

func TestTenantNetworksList(t *testing.T) {
	client, err := clients.NewComputeV2Client()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	allPages, err := tenantnetworks.List(client).AllPages()
	if err != nil {
		t.Fatalf("Unable to list networks: %v", err)
	}

	allTenantNetworks, err := tenantnetworks.ExtractNetworks(allPages)
	if err != nil {
		t.Fatalf("Unable to list networks: %v", err)
	}

	for _, network := range allTenantNetworks {
		PrintTenantNetwork(t, &network)
	}
}

func TestTenantNetworksGet(t *testing.T) {
	choices, err := clients.AcceptanceTestChoicesFromEnv()
	if err != nil {
		t.Fatal(err)
	}

	client, err := clients.NewComputeV2Client()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	networkID, err := GetNetworkIDFromTenantNetworks(t, client, choices.NetworkName)
	if err != nil {
		t.Fatal(err)
	}

	network, err := tenantnetworks.Get(client, networkID).Extract()
	if err != nil {
		t.Fatalf("Unable to get network %s: %v", networkID, err)
	}

	PrintTenantNetwork(t, network)
}
