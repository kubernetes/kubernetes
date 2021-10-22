// +build acceptance networking vpnaas

package vpnaas

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	networks "github.com/gophercloud/gophercloud/acceptance/openstack/networking/v2"
	layer3 "github.com/gophercloud/gophercloud/acceptance/openstack/networking/v2/extensions/layer3"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/layer3/routers"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/vpnaas/siteconnections"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestConnectionList(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	th.AssertNoErr(t, err)

	allPages, err := siteconnections.List(client, nil).AllPages()
	th.AssertNoErr(t, err)

	allConnections, err := siteconnections.ExtractConnections(allPages)
	th.AssertNoErr(t, err)

	for _, connection := range allConnections {
		tools.PrintResource(t, connection)
	}
}

func TestConnectionCRUD(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	th.AssertNoErr(t, err)

	// Create Network
	network, err := networks.CreateNetwork(t, client)
	th.AssertNoErr(t, err)
	defer networks.DeleteNetwork(t, client, network.ID)

	// Create Subnet
	subnet, err := networks.CreateSubnet(t, client, network.ID)
	th.AssertNoErr(t, err)
	defer networks.DeleteSubnet(t, client, subnet.ID)

	router, err := layer3.CreateExternalRouter(t, client)
	th.AssertNoErr(t, err)
	defer layer3.DeleteRouter(t, client, router.ID)

	// Link router and subnet
	aiOpts := routers.AddInterfaceOpts{
		SubnetID: subnet.ID,
	}

	_, err = routers.AddInterface(client, router.ID, aiOpts).Extract()
	th.AssertNoErr(t, err)
	defer func() {
		riOpts := routers.RemoveInterfaceOpts{
			SubnetID: subnet.ID,
		}
		routers.RemoveInterface(client, router.ID, riOpts)
	}()

	// Create all needed resources for the connection
	service, err := CreateService(t, client, router.ID)
	th.AssertNoErr(t, err)
	defer DeleteService(t, client, service.ID)

	ikepolicy, err := CreateIKEPolicy(t, client)
	th.AssertNoErr(t, err)
	defer DeleteIKEPolicy(t, client, ikepolicy.ID)

	ipsecpolicy, err := CreateIPSecPolicy(t, client)
	th.AssertNoErr(t, err)
	defer DeleteIPSecPolicy(t, client, ipsecpolicy.ID)

	peerEPGroup, err := CreateEndpointGroup(t, client)
	th.AssertNoErr(t, err)
	defer DeleteEndpointGroup(t, client, peerEPGroup.ID)

	localEPGroup, err := CreateEndpointGroupWithSubnet(t, client, subnet.ID)
	th.AssertNoErr(t, err)
	defer DeleteEndpointGroup(t, client, localEPGroup.ID)

	conn, err := CreateSiteConnection(t, client, ikepolicy.ID, ipsecpolicy.ID, service.ID, peerEPGroup.ID, localEPGroup.ID)
	th.AssertNoErr(t, err)
	defer DeleteSiteConnection(t, client, conn.ID)

	newConnection, err := siteconnections.Get(client, conn.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, conn)
	tools.PrintResource(t, newConnection)
}
