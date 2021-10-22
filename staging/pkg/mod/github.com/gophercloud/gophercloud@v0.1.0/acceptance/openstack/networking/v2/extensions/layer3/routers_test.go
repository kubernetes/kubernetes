// +build acceptance networking layer3 router

package layer3

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	networking "github.com/gophercloud/gophercloud/acceptance/openstack/networking/v2"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/layer3/routers"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestLayer3RouterCreateDelete(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	th.AssertNoErr(t, err)

	network, err := networking.CreateNetwork(t, client)
	th.AssertNoErr(t, err)
	defer networking.DeleteNetwork(t, client, network.ID)

	router, err := CreateRouter(t, client, network.ID)
	th.AssertNoErr(t, err)
	defer DeleteRouter(t, client, router.ID)

	tools.PrintResource(t, router)

	newName := tools.RandomString("TESTACC-", 8)
	newDescription := ""
	updateOpts := routers.UpdateOpts{
		Name:        newName,
		Description: &newDescription,
	}

	_, err = routers.Update(client, router.ID, updateOpts).Extract()
	th.AssertNoErr(t, err)

	newRouter, err := routers.Get(client, router.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newRouter)
	th.AssertEquals(t, newRouter.Name, newName)
	th.AssertEquals(t, newRouter.Description, newDescription)

	listOpts := routers.ListOpts{}
	allPages, err := routers.List(client, listOpts).AllPages()
	th.AssertNoErr(t, err)

	allRouters, err := routers.ExtractRouters(allPages)
	th.AssertNoErr(t, err)

	var found bool
	for _, router := range allRouters {
		if router.ID == newRouter.ID {
			found = true
		}
	}

	th.AssertEquals(t, found, true)
}

func TestLayer3ExternalRouterCreateDelete(t *testing.T) {
	clients.RequireAdmin(t)

	client, err := clients.NewNetworkV2Client()
	th.AssertNoErr(t, err)

	router, err := CreateExternalRouter(t, client)
	th.AssertNoErr(t, err)
	defer DeleteRouter(t, client, router.ID)

	tools.PrintResource(t, router)

	efi := []routers.ExternalFixedIP{}
	for _, extIP := range router.GatewayInfo.ExternalFixedIPs {
		efi = append(efi,
			routers.ExternalFixedIP{
				IPAddress: extIP.IPAddress,
				SubnetID:  extIP.SubnetID,
			},
		)
	}
	// Add a new external router IP
	efi = append(efi,
		routers.ExternalFixedIP{
			SubnetID: router.GatewayInfo.ExternalFixedIPs[0].SubnetID,
		},
	)

	enableSNAT := true
	gatewayInfo := routers.GatewayInfo{
		NetworkID:        router.GatewayInfo.NetworkID,
		EnableSNAT:       &enableSNAT,
		ExternalFixedIPs: efi,
	}

	newName := tools.RandomString("TESTACC-", 8)
	newDescription := ""
	updateOpts := routers.UpdateOpts{
		Name:        newName,
		Description: &newDescription,
		GatewayInfo: &gatewayInfo,
	}

	_, err = routers.Update(client, router.ID, updateOpts).Extract()
	th.AssertNoErr(t, err)

	newRouter, err := routers.Get(client, router.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newRouter)
	th.AssertEquals(t, newRouter.Name, newName)
	th.AssertEquals(t, newRouter.Description, newDescription)
	th.AssertEquals(t, *newRouter.GatewayInfo.EnableSNAT, enableSNAT)
	th.AssertDeepEquals(t, newRouter.GatewayInfo.ExternalFixedIPs, efi)

	// Test Gateway removal
	updateOpts = routers.UpdateOpts{
		GatewayInfo: &routers.GatewayInfo{},
	}

	newRouter, err = routers.Update(client, router.ID, updateOpts).Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, newRouter.GatewayInfo, routers.GatewayInfo{})
}

func TestLayer3RouterInterface(t *testing.T) {
	clients.RequireAdmin(t)

	client, err := clients.NewNetworkV2Client()
	th.AssertNoErr(t, err)

	// Create Network
	network, err := networking.CreateNetwork(t, client)
	th.AssertNoErr(t, err)
	defer networking.DeleteNetwork(t, client, network.ID)

	subnet, err := networking.CreateSubnet(t, client, network.ID)
	th.AssertNoErr(t, err)
	defer networking.DeleteSubnet(t, client, subnet.ID)

	tools.PrintResource(t, subnet)

	router, err := CreateExternalRouter(t, client)
	th.AssertNoErr(t, err)
	defer DeleteRouter(t, client, router.ID)

	aiOpts := routers.AddInterfaceOpts{
		SubnetID: subnet.ID,
	}

	iface, err := routers.AddInterface(client, router.ID, aiOpts).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, router)
	tools.PrintResource(t, iface)

	riOpts := routers.RemoveInterfaceOpts{
		SubnetID: subnet.ID,
	}

	_, err = routers.RemoveInterface(client, router.ID, riOpts).Extract()
	th.AssertNoErr(t, err)
}
