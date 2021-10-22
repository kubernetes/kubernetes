// +build acceptance networking

package portsbinding

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	networking "github.com/gophercloud/gophercloud/acceptance/openstack/networking/v2"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/portsbinding"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/ports"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestPortsbindingCRUD(t *testing.T) {
	clients.RequireAdmin(t)

	client, err := clients.NewNetworkV2Client()
	th.AssertNoErr(t, err)

	// Create Network
	network, err := networking.CreateNetwork(t, client)
	th.AssertNoErr(t, err)
	defer networking.DeleteNetwork(t, client, network.ID)

	// Create Subnet
	subnet, err := networking.CreateSubnet(t, client, network.ID)
	th.AssertNoErr(t, err)
	defer networking.DeleteSubnet(t, client, subnet.ID)

	// Define a host
	hostID := "localhost"
	profile := map[string]interface{}{"foo": "bar"}

	// Create port
	port, err := CreatePortsbinding(t, client, network.ID, subnet.ID, hostID, profile)
	th.AssertNoErr(t, err)
	defer networking.DeletePort(t, client, port.ID)

	tools.PrintResource(t, port)
	th.AssertEquals(t, port.HostID, hostID)
	th.AssertEquals(t, port.VNICType, "normal")
	th.AssertDeepEquals(t, port.Profile, profile)

	// Update port
	newPortName := ""
	newPortDescription := ""
	newHostID := "127.0.0.1"
	newProfile := map[string]interface{}{}
	updateOpts := ports.UpdateOpts{
		Name:        &newPortName,
		Description: &newPortDescription,
	}

	var finalUpdateOpts ports.UpdateOptsBuilder

	finalUpdateOpts = portsbinding.UpdateOptsExt{
		UpdateOptsBuilder: updateOpts,
		HostID:            &newHostID,
		VNICType:          "baremetal",
		Profile:           newProfile,
	}

	var newPort PortWithBindingExt

	_, err = ports.Update(client, port.ID, finalUpdateOpts).Extract()
	th.AssertNoErr(t, err)

	// Read the updated port
	err = ports.Get(client, port.ID).ExtractInto(&newPort)
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newPort)
	th.AssertEquals(t, newPort.Description, newPortName)
	th.AssertEquals(t, newPort.Description, newPortDescription)
	th.AssertEquals(t, newPort.HostID, newHostID)
	th.AssertEquals(t, newPort.VNICType, "baremetal")
	th.AssertDeepEquals(t, newPort.Profile, newProfile)
}
