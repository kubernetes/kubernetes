package mtu

import (
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/mtu"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/networks"
	th "github.com/gophercloud/gophercloud/testhelper"
)

type NetworkMTU struct {
	networks.Network
	mtu.NetworkMTUExt
}

// CreateNetworkWithMTU will create a network with custom MTU. An error will be
// returned if the creation failed.
func CreateNetworkWithMTU(t *testing.T, client *gophercloud.ServiceClient, networkMTU *int) (*NetworkMTU, error) {
	networkName := tools.RandomString("TESTACC-", 8)
	networkDescription := tools.RandomString("TESTACC-DESC-", 8)

	t.Logf("Attempting to create a network with custom MTU: %s", networkName)

	adminStateUp := true

	var createOpts networks.CreateOptsBuilder
	createOpts = networks.CreateOpts{
		Name:         networkName,
		Description:  networkDescription,
		AdminStateUp: &adminStateUp,
	}

	if *networkMTU > 0 {
		createOpts = mtu.CreateOptsExt{
			CreateOptsBuilder: createOpts,
			MTU:               *networkMTU,
		}
	}

	var network NetworkMTU

	err := networks.Create(client, createOpts).ExtractInto(&network)
	if err != nil {
		return &network, err
	}

	t.Logf("Created a network with custom MTU: %s", networkName)

	th.AssertEquals(t, network.Name, networkName)
	th.AssertEquals(t, network.Description, networkDescription)
	th.AssertEquals(t, network.AdminStateUp, adminStateUp)
	if *networkMTU > 0 {
		th.AssertEquals(t, network.MTU, *networkMTU)
	} else {
		*networkMTU = network.MTU
	}

	return &network, nil
}
