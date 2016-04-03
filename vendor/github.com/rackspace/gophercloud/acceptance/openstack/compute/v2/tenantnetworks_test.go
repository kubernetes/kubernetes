// +build acceptance compute servers

package v2

import (
	"os"
	"testing"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/acceptance/tools"
	"github.com/rackspace/gophercloud/openstack/compute/v2/extensions/tenantnetworks"
	"github.com/rackspace/gophercloud/openstack/compute/v2/servers"
	th "github.com/rackspace/gophercloud/testhelper"
)

func getNetworkID(t *testing.T, client *gophercloud.ServiceClient, networkName string) (string, error) {
	allPages, err := tenantnetworks.List(client).AllPages()
	if err != nil {
		t.Fatalf("Unable to list networks: %v", err)
	}

	networkList, err := tenantnetworks.ExtractNetworks(allPages)
	if err != nil {
		t.Fatalf("Unable to list networks: %v", err)
	}

	networkID := ""
	for _, network := range networkList {
		t.Logf("Network: %v", network)
		if network.Name == networkName {
			networkID = network.ID
		}
	}

	t.Logf("Found network ID for %s: %s\n", networkName, networkID)

	return networkID, nil
}

func createNetworkServer(t *testing.T, client *gophercloud.ServiceClient, choices *ComputeChoices, networkID string) (*servers.Server, error) {
	if testing.Short() {
		t.Skip("Skipping test that requires server creation in short mode.")
	}

	name := tools.RandomString("ACPTTEST", 16)
	t.Logf("Attempting to create server: %s\n", name)

	pwd := tools.MakeNewPassword("")

	networks := make([]servers.Network, 1)
	networks[0] = servers.Network{
		UUID: networkID,
	}

	server, err := servers.Create(client, servers.CreateOpts{
		Name:      name,
		FlavorRef: choices.FlavorID,
		ImageRef:  choices.ImageID,
		AdminPass: pwd,
		Networks:  networks,
	}).Extract()
	if err != nil {
		t.Fatalf("Unable to create server: %v", err)
	}

	th.AssertEquals(t, pwd, server.AdminPass)

	return server, err
}

func TestTenantNetworks(t *testing.T) {
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

	networkID, err := getNetworkID(t, client, networkName)
	if err != nil {
		t.Fatalf("Unable to get network ID: %v", err)
	}

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

	allPages, err := tenantnetworks.List(client).AllPages()
	allNetworks, err := tenantnetworks.ExtractNetworks(allPages)
	th.AssertNoErr(t, err)
	t.Logf("Retrieved all %d networks: %+v", len(allNetworks), allNetworks)
}
