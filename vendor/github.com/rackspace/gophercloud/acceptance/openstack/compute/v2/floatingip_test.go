// +build acceptance compute servers

package v2

import (
	"os"
	"testing"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/acceptance/tools"
	"github.com/rackspace/gophercloud/openstack/compute/v2/extensions/floatingip"
	"github.com/rackspace/gophercloud/openstack/compute/v2/servers"
	th "github.com/rackspace/gophercloud/testhelper"
)

func createFIPServer(t *testing.T, client *gophercloud.ServiceClient, choices *ComputeChoices) (*servers.Server, error) {
	if testing.Short() {
		t.Skip("Skipping test that requires server creation in short mode.")
	}

	name := tools.RandomString("ACPTTEST", 16)
	t.Logf("Attempting to create server: %s\n", name)

	pwd := tools.MakeNewPassword("")

	server, err := servers.Create(client, servers.CreateOpts{
		Name:      name,
		FlavorRef: choices.FlavorID,
		ImageRef:  choices.ImageID,
		AdminPass: pwd,
	}).Extract()
	if err != nil {
		t.Fatalf("Unable to create server: %v", err)
	}

	th.AssertEquals(t, pwd, server.AdminPass)

	return server, err
}

func createFloatingIP(t *testing.T, client *gophercloud.ServiceClient) (*floatingip.FloatingIP, error) {
	pool := os.Getenv("OS_POOL_NAME")
	fip, err := floatingip.Create(client, &floatingip.CreateOpts{
		Pool: pool,
	}).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Obtained Floating IP: %v", fip.IP)

	return fip, err
}

func associateFloatingIPDeprecated(t *testing.T, client *gophercloud.ServiceClient, serverId string, fip *floatingip.FloatingIP) {
	// This form works, but is considered deprecated.
	// See associateFloatingIP or associateFloatingIPFixed
	err := floatingip.Associate(client, serverId, fip.IP).ExtractErr()
	th.AssertNoErr(t, err)
	t.Logf("Associated floating IP %v from instance %v", fip.IP, serverId)
	defer func() {
		err = floatingip.Disassociate(client, serverId, fip.IP).ExtractErr()
		th.AssertNoErr(t, err)
		t.Logf("Disassociated floating IP %v from instance %v", fip.IP, serverId)
	}()
	floatingIp, err := floatingip.Get(client, fip.ID).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Floating IP %v is associated with Fixed IP %v", fip.IP, floatingIp.FixedIP)
}

func associateFloatingIP(t *testing.T, client *gophercloud.ServiceClient, serverId string, fip *floatingip.FloatingIP) {
	associateOpts := floatingip.AssociateOpts{
		ServerID:   serverId,
		FloatingIP: fip.IP,
	}

	err := floatingip.AssociateInstance(client, associateOpts).ExtractErr()
	th.AssertNoErr(t, err)
	t.Logf("Associated floating IP %v from instance %v", fip.IP, serverId)
	defer func() {
		err = floatingip.DisassociateInstance(client, associateOpts).ExtractErr()
		th.AssertNoErr(t, err)
		t.Logf("Disassociated floating IP %v from instance %v", fip.IP, serverId)
	}()
	floatingIp, err := floatingip.Get(client, fip.ID).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Floating IP %v is associated with Fixed IP %v", fip.IP, floatingIp.FixedIP)
}

func associateFloatingIPFixed(t *testing.T, client *gophercloud.ServiceClient, serverId string, fip *floatingip.FloatingIP) {

	network := os.Getenv("OS_NETWORK_NAME")
	server, err := servers.Get(client, serverId).Extract()
	if err != nil {
		t.Fatalf("%s", err)
	}

	var fixedIP string
	for _, networkAddresses := range server.Addresses[network].([]interface{}) {
		address := networkAddresses.(map[string]interface{})
		if address["OS-EXT-IPS:type"] == "fixed" {
			if address["version"].(float64) == 4 {
				fixedIP = address["addr"].(string)
			}
		}
	}

	associateOpts := floatingip.AssociateOpts{
		ServerID:   serverId,
		FloatingIP: fip.IP,
		FixedIP:    fixedIP,
	}

	err = floatingip.AssociateInstance(client, associateOpts).ExtractErr()
	th.AssertNoErr(t, err)
	t.Logf("Associated floating IP %v from instance %v with Fixed IP %v", fip.IP, serverId, fixedIP)
	defer func() {
		err = floatingip.DisassociateInstance(client, associateOpts).ExtractErr()
		th.AssertNoErr(t, err)
		t.Logf("Disassociated floating IP %v from instance %v with Fixed IP %v", fip.IP, serverId, fixedIP)
	}()
	floatingIp, err := floatingip.Get(client, fip.ID).Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, floatingIp.FixedIP, fixedIP)
	t.Logf("Floating IP %v is associated with Fixed IP %v", fip.IP, floatingIp.FixedIP)
}

func TestFloatingIP(t *testing.T) {
	pool := os.Getenv("OS_POOL_NAME")
	if pool == "" {
		t.Fatalf("OS_POOL_NAME must be set")
	}

	choices, err := ComputeChoicesFromEnv()
	if err != nil {
		t.Fatal(err)
	}

	client, err := newClient()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	server, err := createFIPServer(t, client, choices)
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

	fip, err := createFloatingIP(t, client)
	if err != nil {
		t.Fatalf("Unable to create floating IP: %v", err)
	}
	defer func() {
		err = floatingip.Delete(client, fip.ID).ExtractErr()
		th.AssertNoErr(t, err)
		t.Logf("Floating IP deleted.")
	}()

	associateFloatingIPDeprecated(t, client, server.ID, fip)
	associateFloatingIP(t, client, server.ID, fip)
	associateFloatingIPFixed(t, client, server.ID, fip)

}
