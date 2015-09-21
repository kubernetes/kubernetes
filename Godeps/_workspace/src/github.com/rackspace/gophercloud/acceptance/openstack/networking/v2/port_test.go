// +build acceptance networking

package v2

import (
	"testing"

	"github.com/rackspace/gophercloud/openstack/networking/v2/networks"
	"github.com/rackspace/gophercloud/openstack/networking/v2/ports"
	"github.com/rackspace/gophercloud/openstack/networking/v2/subnets"
	"github.com/rackspace/gophercloud/pagination"
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
	th.AssertEquals(t, p.ID, portID)

	// Update port
	p, err = ports.Update(Client, portID, ports.UpdateOpts{Name: "new_port_name"}).Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, p.Name, "new_port_name")

	// Delete port
	res := ports.Delete(Client, portID)
	th.AssertNoErr(t, res.Err)
}

func createPort(t *testing.T, networkID, subnetID string) string {
	enable := false
	opts := ports.CreateOpts{
		NetworkID:    networkID,
		Name:         "my_port",
		AdminStateUp: &enable,
		FixedIPs:     []ports.IP{ports.IP{SubnetID: subnetID}},
	}
	p, err := ports.Create(Client, opts).Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, p.NetworkID, networkID)
	th.AssertEquals(t, p.Name, "my_port")
	th.AssertEquals(t, p.AdminStateUp, false)

	return p.ID
}

func listPorts(t *testing.T) {
	count := 0
	pager := ports.List(Client, ports.ListOpts{})
	err := pager.EachPage(func(page pagination.Page) (bool, error) {
		count++
		t.Logf("--- Page ---")

		portList, err := ports.ExtractPorts(page)
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
	res, err := networks.Create(Client, networks.CreateOpts{Name: "tmp_network", AdminStateUp: networks.Up}).Extract()
	return res.ID, err
}

func createSubnet(networkID string) (string, error) {
	s, err := subnets.Create(Client, subnets.CreateOpts{
		NetworkID:  networkID,
		CIDR:       "192.168.199.0/24",
		IPVersion:  subnets.IPv4,
		Name:       "my_subnet",
		EnableDHCP: subnets.Down,
	}).Extract()
	return s.ID, err
}

func TestPortBatchCreate(t *testing.T) {
	// todo
}
