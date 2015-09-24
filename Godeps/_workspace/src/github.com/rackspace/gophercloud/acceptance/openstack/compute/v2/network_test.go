// +build acceptance compute servers

package v2

import (
	"os"
	"testing"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/openstack/compute/v2/extensions/networks"
	"github.com/rackspace/gophercloud/openstack/compute/v2/servers"
	th "github.com/rackspace/gophercloud/testhelper"
)

func getNetworkIDFromNetworkExtension(t *testing.T, client *gophercloud.ServiceClient, networkName string) (string, error) {
	allPages, err := networks.List(client).AllPages()
	if err != nil {
		t.Fatalf("Unable to list networks: %v", err)
	}

	networkList, err := networks.ExtractNetworks(allPages)
	if err != nil {
		t.Fatalf("Unable to list networks: %v", err)
	}

	networkID := ""
	for _, network := range networkList {
		t.Logf("Network: %v", network)
		if network.Label == networkName {
			networkID = network.ID
		}
	}

	t.Logf("Found network ID for %s: %s\n", networkName, networkID)

	return networkID, nil
}

func TestNetworks(t *testing.T) {
	networkName := os.Getenv("OS_NETWORK_NAME")
	if networkName == "" {
		t.Fatalf("OS_NETWORK_NAME must be set")
	}

	choices, err := ComputeChoicesFromEnv()
	if err != nil {
		t.Fatal(err)
	}

	client, err := newClient()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	networkID, err := getNetworkIDFromNetworkExtension(t, client, networkName)
	if err != nil {
		t.Fatalf("Unable to get network ID: %v", err)
	}

	// createNetworkServer is defined in tenantnetworks_test.go
	server, err := createNetworkServer(t, client, choices, networkID)
	if err != nil {
		t.Fatalf("Unable to create server: %v", err)
	}
	defer func() {
		servers.Delete(client, server.ID)
		t.Logf("Server deleted.")
	}()

	if err = waitForStatus(client, server, "ACTIVE"); err != nil {
		t.Fatalf("Unable to wait for server: %v", err)
	}

	allPages, err := networks.List(client).AllPages()
	allNetworks, err := networks.ExtractNetworks(allPages)
	th.AssertNoErr(t, err)
	t.Logf("Retrieved all %d networks: %+v", len(allNetworks), allNetworks)
}
