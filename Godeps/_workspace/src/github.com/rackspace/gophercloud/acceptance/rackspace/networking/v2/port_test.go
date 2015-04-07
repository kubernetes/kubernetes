// +build acceptance networking

package v2

import (
	"testing"

	osNetworks "github.com/rackspace/gophercloud/openstack/networking/v2/networks"
	osPorts "github.com/rackspace/gophercloud/openstack/networking/v2/ports"
	osSubnets "github.com/rackspace/gophercloud/openstack/networking/v2/subnets"
	"github.com/rackspace/gophercloud/pagination"
	"github.com/rackspace/gophercloud/rackspace/networking/v2/networks"
	"github.com/rackspace/gophercloud/rackspace/networking/v2/ports"
	"github.com/rackspace/gophercloud/rackspace/networking/v2/subnets"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestPortCRUD(t *testing.T) {
	Setup(t)
	defer Teardown()

	// Setup network
	t.Log("Setting up network")
	networkID, err := createNetwork()
	th.AssertNoErr(t, err)
	defer networks.Delete(Client, networkID)

	// Setup subnet
	t.Logf("Setting up subnet on network %s", networkID)
	subnetID, err := createSubnet(networkID)
	th.AssertNoErr(t, err)
	defer subnets.Delete(Client, subnetID)

	// Create port
	t.Logf("Create port based on subnet %s", subnetID)
	portID := createPort(t, networkID, subnetID)

	// List ports
	t.Logf("Listing all ports")
	listPorts(t)

	// Get port
	if portID == "" {
		t.Fatalf("In order to retrieve a port, the portID must be set")
	}
	p, err := ports.Get(Client, portID).Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, portID, p.ID)

	// Update port
	p, err = ports.Update(Client, portID, osPorts.UpdateOpts{Name: "new_port_name"}).Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, "new_port_name", p.Name)

	// Delete port
	res := ports.Delete(Client, portID)
	th.AssertNoErr(t, res.Err)
}

func createPort(t *testing.T, networkID, subnetID string) string {
	enable := true
	opts := osPorts.CreateOpts{
		NetworkID:    networkID,
		Name:         "my_port",
		AdminStateUp: &enable,
		FixedIPs:     []osPorts.IP{osPorts.IP{SubnetID: subnetID}},
	}
	p, err := ports.Create(Client, opts).Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, networkID, p.NetworkID)
	th.AssertEquals(t, "my_port", p.Name)
	th.AssertEquals(t, true, p.AdminStateUp)

	return p.ID
}

func listPorts(t *testing.T) {
	count := 0
	pager := ports.List(Client, osPorts.ListOpts{})
	err := pager.EachPage(func(page pagination.Page) (bool, error) {
		count++
		t.Logf("--- Page ---")

		portList, err := osPorts.ExtractPorts(page)
		th.AssertNoErr(t, err)

		for _, p := range portList {
			t.Logf("Port: ID [%s] Name [%s] Status [%s] MAC addr [%s] Fixed IPs [%#v] Security groups [%#v]",
				p.ID, p.Name, p.Status, p.MACAddress, p.FixedIPs, p.SecurityGroups)
		}

		return true, nil
	})

	th.CheckNoErr(t, err)

	if count == 0 {
		t.Logf("No pages were iterated over when listing ports")
	}
}

func createNetwork() (string, error) {
	res, err := networks.Create(Client, osNetworks.CreateOpts{Name: "tmp_network", AdminStateUp: osNetworks.Up}).Extract()
	return res.ID, err
}

func createSubnet(networkID string) (string, error) {
	s, err := subnets.Create(Client, osSubnets.CreateOpts{
		NetworkID:  networkID,
		CIDR:       "192.168.199.0/24",
		IPVersion:  osSubnets.IPv4,
		Name:       "my_subnet",
		EnableDHCP: osSubnets.Down,
	}).Extract()
	return s.ID, err
}
