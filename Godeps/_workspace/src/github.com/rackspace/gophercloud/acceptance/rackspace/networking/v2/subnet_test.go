// +build acceptance networking

package v2

import (
	"testing"

	osNetworks "github.com/rackspace/gophercloud/openstack/networking/v2/networks"
	osSubnets "github.com/rackspace/gophercloud/openstack/networking/v2/subnets"
	"github.com/rackspace/gophercloud/pagination"
	"github.com/rackspace/gophercloud/rackspace/networking/v2/networks"
	"github.com/rackspace/gophercloud/rackspace/networking/v2/subnets"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestListSubnets(t *testing.T) {
	Setup(t)
	defer Teardown()

	pager := subnets.List(Client, osSubnets.ListOpts{Limit: 2})
	err := pager.EachPage(func(page pagination.Page) (bool, error) {
		t.Logf("--- Page ---")

		subnetList, err := osSubnets.ExtractSubnets(page)
		th.AssertNoErr(t, err)

		for _, s := range subnetList {
			t.Logf("Subnet: ID [%s] Name [%s] IP Version [%d] CIDR [%s] GatewayIP [%s]",
				s.ID, s.Name, s.IPVersion, s.CIDR, s.GatewayIP)
		}

		return true, nil
	})
	th.CheckNoErr(t, err)
}

func TestSubnetCRUD(t *testing.T) {
	Setup(t)
	defer Teardown()

	// Setup network
	t.Log("Setting up network")
	n, err := networks.Create(Client, osNetworks.CreateOpts{Name: "tmp_network", AdminStateUp: osNetworks.Up}).Extract()
	th.AssertNoErr(t, err)
	networkID := n.ID
	defer networks.Delete(Client, networkID)

	// Create subnet
	t.Log("Create subnet")
	enable := false
	opts := osSubnets.CreateOpts{
		NetworkID:  networkID,
		CIDR:       "192.168.199.0/24",
		IPVersion:  osSubnets.IPv4,
		Name:       "my_subnet",
		EnableDHCP: &enable,
	}
	s, err := subnets.Create(Client, opts).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, networkID, s.NetworkID)
	th.AssertEquals(t, "192.168.199.0/24", s.CIDR)
	th.AssertEquals(t, 4, s.IPVersion)
	th.AssertEquals(t, "my_subnet", s.Name)
	th.AssertEquals(t, false, s.EnableDHCP)
	subnetID := s.ID

	// Get subnet
	t.Log("Getting subnet")
	s, err = subnets.Get(Client, subnetID).Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, subnetID, s.ID)

	// Update subnet
	t.Log("Update subnet")
	s, err = subnets.Update(Client, subnetID, osSubnets.UpdateOpts{Name: "new_subnet_name"}).Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, "new_subnet_name", s.Name)

	// Delete subnet
	t.Log("Delete subnet")
	res := subnets.Delete(Client, subnetID)
	th.AssertNoErr(t, res.Err)
}
