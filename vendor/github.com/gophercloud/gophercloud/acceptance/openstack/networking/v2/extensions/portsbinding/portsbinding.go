package portsbinding

import (
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/portsbinding"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/ports"
)

// PortWithBindingExt represents a port with the binding fields
type PortWithBindingExt struct {
	ports.Port
	portsbinding.PortsBindingExt
}

// CreatePortsbinding will create a port on the specified subnet. An error will be
// returned if the port could not be created.
func CreatePortsbinding(t *testing.T, client *gophercloud.ServiceClient, networkID, subnetID, hostID string) (PortWithBindingExt, error) {
	portName := tools.RandomString("TESTACC-", 8)
	iFalse := false

	t.Logf("Attempting to create port: %s", portName)

	portCreateOpts := ports.CreateOpts{
		NetworkID:    networkID,
		Name:         portName,
		AdminStateUp: &iFalse,
		FixedIPs:     []ports.IP{ports.IP{SubnetID: subnetID}},
	}

	createOpts := portsbinding.CreateOptsExt{
		CreateOptsBuilder: portCreateOpts,
		HostID:            hostID,
	}

	var s PortWithBindingExt

	err := ports.Create(client, createOpts).ExtractInto(&s)
	if err != nil {
		return s, err
	}

	t.Logf("Successfully created port: %s", portName)

	return s, nil
}
