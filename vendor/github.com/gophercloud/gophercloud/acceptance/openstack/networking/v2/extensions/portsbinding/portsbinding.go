package portsbinding

import (
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/portsbinding"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/ports"
)

// CreatePortsbinding will create a port on the specified subnet. An error will be
// returned if the port could not be created.
func CreatePortsbinding(t *testing.T, client *gophercloud.ServiceClient, networkID, subnetID, hostID string) (*portsbinding.Port, error) {
	portName := tools.RandomString("TESTACC-", 8)
	iFalse := false

	t.Logf("Attempting to create port: %s", portName)

	createOpts := portsbinding.CreateOpts{
		CreateOptsBuilder: ports.CreateOpts{
			NetworkID:    networkID,
			Name:         portName,
			AdminStateUp: &iFalse,
			FixedIPs:     []ports.IP{ports.IP{SubnetID: subnetID}},
		},
		HostID: hostID,
	}

	port, err := portsbinding.Create(client, createOpts).Extract()
	if err != nil {
		return port, err
	}

	t.Logf("Successfully created port: %s", portName)

	return port, nil
}

// PrintPortsbinging will print a port and all of its attributes.
func PrintPortsbinding(t *testing.T, port *portsbinding.Port) {
	t.Logf("ID: %s", port.ID)
	t.Logf("NetworkID: %s", port.NetworkID)
	t.Logf("Name: %s", port.Name)
	t.Logf("AdminStateUp: %t", port.AdminStateUp)
	t.Logf("Status: %s", port.Status)
	t.Logf("MACAddress: %s", port.MACAddress)
	t.Logf("FixedIPs: %s", port.FixedIPs)
	t.Logf("TenantID: %s", port.TenantID)
	t.Logf("DeviceOwner: %s", port.DeviceOwner)
	t.Logf("SecurityGroups: %s", port.SecurityGroups)
	t.Logf("DeviceID: %s", port.DeviceID)
	t.Logf("DeviceOwner: %s", port.DeviceOwner)
	t.Logf("AllowedAddressPairs: %s", port.AllowedAddressPairs)
	t.Logf("HostID: %s", port.HostID)
	t.Logf("VNICType: %s", port.VNICType)
}
