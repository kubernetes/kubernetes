// +build acceptance networking

package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	extensions "github.com/gophercloud/gophercloud/acceptance/openstack/networking/v2/extensions"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/extradhcpopts"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/portsecurity"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/ports"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestPortsCRUD(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	th.AssertNoErr(t, err)

	// Create Network
	network, err := CreateNetwork(t, client)
	th.AssertNoErr(t, err)
	defer DeleteNetwork(t, client, network.ID)

	// Create Subnet
	subnet, err := CreateSubnet(t, client, network.ID)
	th.AssertNoErr(t, err)
	defer DeleteSubnet(t, client, subnet.ID)

	// Create port
	port, err := CreatePort(t, client, network.ID, subnet.ID)
	th.AssertNoErr(t, err)
	defer DeletePort(t, client, port.ID)

	if len(port.SecurityGroups) != 1 {
		t.Logf("WARNING: Port did not have a default security group applied")
	}

	tools.PrintResource(t, port)

	// Update port
	newPortName := ""
	newPortDescription := ""
	updateOpts := ports.UpdateOpts{
		Name:        &newPortName,
		Description: &newPortDescription,
	}
	newPort, err := ports.Update(client, port.ID, updateOpts).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newPort)

	th.AssertEquals(t, newPort.Name, newPortName)
	th.AssertEquals(t, newPort.Description, newPortDescription)

	allPages, err := ports.List(client, nil).AllPages()
	th.AssertNoErr(t, err)

	allPorts, err := ports.ExtractPorts(allPages)
	th.AssertNoErr(t, err)

	var found bool
	for _, port := range allPorts {
		if port.ID == newPort.ID {
			found = true
		}
	}

	th.AssertEquals(t, found, true)
}

func TestPortsRemoveSecurityGroups(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	th.AssertNoErr(t, err)

	// Create Network
	network, err := CreateNetwork(t, client)
	th.AssertNoErr(t, err)
	defer DeleteNetwork(t, client, network.ID)

	// Create Subnet
	subnet, err := CreateSubnet(t, client, network.ID)
	th.AssertNoErr(t, err)
	defer DeleteSubnet(t, client, subnet.ID)

	// Create port
	port, err := CreatePort(t, client, network.ID, subnet.ID)
	th.AssertNoErr(t, err)
	defer DeletePort(t, client, port.ID)

	tools.PrintResource(t, port)

	// Create a Security Group
	group, err := extensions.CreateSecurityGroup(t, client)
	th.AssertNoErr(t, err)
	defer extensions.DeleteSecurityGroup(t, client, group.ID)

	// Add the group to the port
	updateOpts := ports.UpdateOpts{
		SecurityGroups: &[]string{group.ID},
	}
	newPort, err := ports.Update(client, port.ID, updateOpts).Extract()
	th.AssertNoErr(t, err)

	// Remove the group
	updateOpts = ports.UpdateOpts{
		SecurityGroups: &[]string{},
	}
	newPort, err = ports.Update(client, port.ID, updateOpts).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newPort)

	if len(newPort.SecurityGroups) > 0 {
		t.Fatalf("Unable to remove security group from port")
	}
}

func TestPortsDontAlterSecurityGroups(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	th.AssertNoErr(t, err)

	// Create Network
	network, err := CreateNetwork(t, client)
	th.AssertNoErr(t, err)
	defer DeleteNetwork(t, client, network.ID)

	// Create Subnet
	subnet, err := CreateSubnet(t, client, network.ID)
	th.AssertNoErr(t, err)
	defer DeleteSubnet(t, client, subnet.ID)

	// Create a Security Group
	group, err := extensions.CreateSecurityGroup(t, client)
	th.AssertNoErr(t, err)
	defer extensions.DeleteSecurityGroup(t, client, group.ID)

	// Create port
	port, err := CreatePort(t, client, network.ID, subnet.ID)
	th.AssertNoErr(t, err)
	defer DeletePort(t, client, port.ID)

	tools.PrintResource(t, port)

	// Add the group to the port
	updateOpts := ports.UpdateOpts{
		SecurityGroups: &[]string{group.ID},
	}
	newPort, err := ports.Update(client, port.ID, updateOpts).Extract()
	th.AssertNoErr(t, err)

	// Update the port again
	var name = "some_port"
	updateOpts = ports.UpdateOpts{
		Name: &name,
	}
	newPort, err = ports.Update(client, port.ID, updateOpts).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newPort)

	if len(newPort.SecurityGroups) == 0 {
		t.Fatalf("Port had security group updated")
	}
}

func TestPortsWithNoSecurityGroup(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	th.AssertNoErr(t, err)

	// Create Network
	network, err := CreateNetwork(t, client)
	th.AssertNoErr(t, err)
	defer DeleteNetwork(t, client, network.ID)

	// Create Subnet
	subnet, err := CreateSubnet(t, client, network.ID)
	th.AssertNoErr(t, err)
	defer DeleteSubnet(t, client, subnet.ID)

	// Create port
	port, err := CreatePortWithNoSecurityGroup(t, client, network.ID, subnet.ID)
	th.AssertNoErr(t, err)
	defer DeletePort(t, client, port.ID)

	tools.PrintResource(t, port)

	if len(port.SecurityGroups) != 0 {
		t.Fatalf("Port was created with security groups")
	}
}

func TestPortsRemoveAddressPair(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	th.AssertNoErr(t, err)

	// Create Network
	network, err := CreateNetwork(t, client)
	th.AssertNoErr(t, err)
	defer DeleteNetwork(t, client, network.ID)

	// Create Subnet
	subnet, err := CreateSubnet(t, client, network.ID)
	th.AssertNoErr(t, err)
	defer DeleteSubnet(t, client, subnet.ID)

	// Create port
	port, err := CreatePort(t, client, network.ID, subnet.ID)
	th.AssertNoErr(t, err)
	defer DeletePort(t, client, port.ID)

	tools.PrintResource(t, port)

	// Add an address pair to the port
	updateOpts := ports.UpdateOpts{
		AllowedAddressPairs: &[]ports.AddressPair{
			ports.AddressPair{IPAddress: "192.168.255.10", MACAddress: "aa:bb:cc:dd:ee:ff"},
		},
	}
	newPort, err := ports.Update(client, port.ID, updateOpts).Extract()
	th.AssertNoErr(t, err)

	// Remove the address pair
	updateOpts = ports.UpdateOpts{
		AllowedAddressPairs: &[]ports.AddressPair{},
	}
	newPort, err = ports.Update(client, port.ID, updateOpts).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newPort)

	if len(newPort.AllowedAddressPairs) > 0 {
		t.Fatalf("Unable to remove the address pair")
	}
}

func TestPortsDontUpdateAllowedAddressPairs(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	th.AssertNoErr(t, err)

	// Create Network
	network, err := CreateNetwork(t, client)
	th.AssertNoErr(t, err)
	defer DeleteNetwork(t, client, network.ID)

	// Create Subnet
	subnet, err := CreateSubnet(t, client, network.ID)
	th.AssertNoErr(t, err)
	defer DeleteSubnet(t, client, subnet.ID)

	// Create port
	port, err := CreatePort(t, client, network.ID, subnet.ID)
	th.AssertNoErr(t, err)
	defer DeletePort(t, client, port.ID)

	tools.PrintResource(t, port)

	// Add an address pair to the port
	updateOpts := ports.UpdateOpts{
		AllowedAddressPairs: &[]ports.AddressPair{
			ports.AddressPair{IPAddress: "192.168.255.10", MACAddress: "aa:bb:cc:dd:ee:ff"},
		},
	}
	newPort, err := ports.Update(client, port.ID, updateOpts).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newPort)

	// Remove the address pair
	var name = "some_port"
	updateOpts = ports.UpdateOpts{
		Name: &name,
	}
	newPort, err = ports.Update(client, port.ID, updateOpts).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newPort)

	if len(newPort.AllowedAddressPairs) == 0 {
		t.Fatalf("Address Pairs were removed")
	}
}

func TestPortsPortSecurityCRUD(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	th.AssertNoErr(t, err)

	// Create Network
	network, err := CreateNetwork(t, client)
	th.AssertNoErr(t, err)
	defer DeleteNetwork(t, client, network.ID)

	// Create Subnet
	subnet, err := CreateSubnet(t, client, network.ID)
	th.AssertNoErr(t, err)
	defer DeleteSubnet(t, client, subnet.ID)

	// Create port
	port, err := CreatePortWithoutPortSecurity(t, client, network.ID, subnet.ID)
	th.AssertNoErr(t, err)
	defer DeletePort(t, client, port.ID)

	var portWithExt struct {
		ports.Port
		portsecurity.PortSecurityExt
	}

	err = ports.Get(client, port.ID).ExtractInto(&portWithExt)
	th.AssertNoErr(t, err)

	tools.PrintResource(t, portWithExt)

	iTrue := true
	portUpdateOpts := ports.UpdateOpts{}
	updateOpts := portsecurity.PortUpdateOptsExt{
		UpdateOptsBuilder:   portUpdateOpts,
		PortSecurityEnabled: &iTrue,
	}

	err = ports.Update(client, port.ID, updateOpts).ExtractInto(&portWithExt)
	th.AssertNoErr(t, err)

	tools.PrintResource(t, portWithExt)
}

func TestPortsWithExtraDHCPOptsCRUD(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	th.AssertNoErr(t, err)

	// Create a Network
	network, err := CreateNetwork(t, client)
	th.AssertNoErr(t, err)
	defer DeleteNetwork(t, client, network.ID)

	// Create a Subnet
	subnet, err := CreateSubnet(t, client, network.ID)
	th.AssertNoErr(t, err)
	defer DeleteSubnet(t, client, subnet.ID)

	// Create a port with extra DHCP options.
	port, err := CreatePortWithExtraDHCPOpts(t, client, network.ID, subnet.ID)
	th.AssertNoErr(t, err)
	defer DeletePort(t, client, port.ID)

	tools.PrintResource(t, port)

	// Update the port with extra DHCP options.
	newPortName := tools.RandomString("TESTACC-", 8)
	portUpdateOpts := ports.UpdateOpts{
		Name: &newPortName,
	}

	existingOpt := port.ExtraDHCPOpts[0]
	newOptValue := "test_value_2"

	updateOpts := extradhcpopts.UpdateOptsExt{
		UpdateOptsBuilder: portUpdateOpts,
		ExtraDHCPOpts: []extradhcpopts.UpdateExtraDHCPOpt{
			{
				OptName:  existingOpt.OptName,
				OptValue: nil,
			},
			{
				OptName:  "test_option_2",
				OptValue: &newOptValue,
			},
		},
	}

	newPort := &PortWithExtraDHCPOpts{}
	err = ports.Update(client, port.ID, updateOpts).ExtractInto(newPort)
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newPort)
}
