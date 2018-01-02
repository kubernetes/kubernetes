// +build acceptance compute servers

package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/floatingips"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/servers"
)

func TestFloatingIPsList(t *testing.T) {
	client, err := clients.NewComputeV2Client()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	allPages, err := floatingips.List(client).AllPages()
	if err != nil {
		t.Fatalf("Unable to retrieve floating IPs: %v", err)
	}

	allFloatingIPs, err := floatingips.ExtractFloatingIPs(allPages)
	if err != nil {
		t.Fatalf("Unable to extract floating IPs: %v", err)
	}

	for _, floatingIP := range allFloatingIPs {
		tools.PrintResource(t, floatingIP)
	}
}

func TestFloatingIPsCreate(t *testing.T) {
	client, err := clients.NewComputeV2Client()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	floatingIP, err := CreateFloatingIP(t, client)
	if err != nil {
		t.Fatalf("Unable to create floating IP: %v", err)
	}
	defer DeleteFloatingIP(t, client, floatingIP)

	tools.PrintResource(t, floatingIP)
}

func TestFloatingIPsAssociate(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping test that requires server creation in short mode.")
	}

	client, err := clients.NewComputeV2Client()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	server, err := CreateServer(t, client)
	if err != nil {
		t.Fatalf("Unable to create server: %v", err)
	}
	defer DeleteServer(t, client, server)

	floatingIP, err := CreateFloatingIP(t, client)
	if err != nil {
		t.Fatalf("Unable to create floating IP: %v", err)
	}
	defer DeleteFloatingIP(t, client, floatingIP)

	tools.PrintResource(t, floatingIP)

	err = AssociateFloatingIP(t, client, floatingIP, server)
	if err != nil {
		t.Fatalf("Unable to associate floating IP %s with server %s: %v", floatingIP.IP, server.ID, err)
	}
	defer DisassociateFloatingIP(t, client, floatingIP, server)

	newFloatingIP, err := floatingips.Get(client, floatingIP.ID).Extract()
	if err != nil {
		t.Fatalf("Unable to get floating IP %s: %v", floatingIP.ID, err)
	}

	t.Logf("Floating IP %s is associated with Fixed IP %s", floatingIP.IP, newFloatingIP.FixedIP)

	tools.PrintResource(t, newFloatingIP)
}

func TestFloatingIPsFixedIPAssociate(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping test that requires server creation in short mode.")
	}

	client, err := clients.NewComputeV2Client()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	choices, err := clients.AcceptanceTestChoicesFromEnv()
	if err != nil {
		t.Fatal(err)
	}

	server, err := CreateServer(t, client)
	if err != nil {
		t.Fatalf("Unable to create server: %v", err)
	}
	defer DeleteServer(t, client, server)

	newServer, err := servers.Get(client, server.ID).Extract()
	if err != nil {
		t.Fatalf("Unable to get server %s: %v", server.ID, err)
	}

	floatingIP, err := CreateFloatingIP(t, client)
	if err != nil {
		t.Fatalf("Unable to create floating IP: %v", err)
	}
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
	if err != nil {
		t.Fatalf("Unable to associate floating IP %s with server %s: %v", floatingIP.IP, newServer.ID, err)
	}
	defer DisassociateFloatingIP(t, client, floatingIP, newServer)

	newFloatingIP, err := floatingips.Get(client, floatingIP.ID).Extract()
	if err != nil {
		t.Fatalf("Unable to get floating IP %s: %v", floatingIP.ID, err)
	}

	t.Logf("Floating IP %s is associated with Fixed IP %s", floatingIP.IP, newFloatingIP.FixedIP)

	tools.PrintResource(t, newFloatingIP)
}
