package layer3

import (
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/layer3/floatingips"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/layer3/routers"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/ports"
)

// CreateFloatingIP creates a floating IP on a given network and port. An error
// will be returned if the creation failed.
func CreateFloatingIP(t *testing.T, client *gophercloud.ServiceClient, networkID, portID string) (*floatingips.FloatingIP, error) {
	t.Logf("Attempting to create floating IP on port: %s", portID)

	createOpts := &floatingips.CreateOpts{
		FloatingNetworkID: networkID,
		PortID:            portID,
	}

	floatingIP, err := floatingips.Create(client, createOpts).Extract()
	if err != nil {
		return floatingIP, err
	}

	t.Logf("Created floating IP.")

	return floatingIP, err
}

// CreateExternalRouter creates a router on the external network. This requires
// the OS_EXTGW_ID environment variable to be set. An error is returned if the
// creation failed.
func CreateExternalRouter(t *testing.T, client *gophercloud.ServiceClient) (*routers.Router, error) {
	var router *routers.Router
	choices, err := clients.AcceptanceTestChoicesFromEnv()
	if err != nil {
		return router, err
	}

	routerName := tools.RandomString("TESTACC-", 8)

	t.Logf("Attempting to create external router: %s", routerName)

	adminStateUp := true
	gatewayInfo := routers.GatewayInfo{
		NetworkID: choices.ExternalNetworkID,
	}

	createOpts := routers.CreateOpts{
		Name:         routerName,
		AdminStateUp: &adminStateUp,
		GatewayInfo:  &gatewayInfo,
	}

	router, err = routers.Create(client, createOpts).Extract()
	if err != nil {
		return router, err
	}

	if err := WaitForRouterToCreate(client, router.ID, 60); err != nil {
		return router, err
	}

	t.Logf("Created router: %s", routerName)

	return router, nil
}

// CreateRouter creates a router on a specified Network ID. An error will be
// returned if the creation failed.
func CreateRouter(t *testing.T, client *gophercloud.ServiceClient, networkID string) (*routers.Router, error) {
	routerName := tools.RandomString("TESTACC-", 8)

	t.Logf("Attempting to create router: %s", routerName)

	adminStateUp := true
	gatewayInfo := routers.GatewayInfo{
		NetworkID: networkID,
	}

	createOpts := routers.CreateOpts{
		Name:         routerName,
		AdminStateUp: &adminStateUp,
		GatewayInfo:  &gatewayInfo,
	}

	router, err := routers.Create(client, createOpts).Extract()
	if err != nil {
		return router, err
	}

	if err := WaitForRouterToCreate(client, router.ID, 60); err != nil {
		return router, err
	}

	t.Logf("Created router: %s", routerName)

	return router, nil
}

// CreateRouterInterface will attach a subnet to a router. An error will be
// returned if the operation fails.
func CreateRouterInterface(t *testing.T, client *gophercloud.ServiceClient, portID, routerID string) (*routers.InterfaceInfo, error) {
	t.Logf("Attempting to add port %s to router %s", portID, routerID)

	aiOpts := routers.AddInterfaceOpts{
		PortID: portID,
	}

	iface, err := routers.AddInterface(client, routerID, aiOpts).Extract()
	if err != nil {
		return iface, err
	}

	if err := WaitForRouterInterfaceToAttach(client, portID, 60); err != nil {
		return iface, err
	}

	t.Logf("Successfully added port %s to router %s", portID, routerID)
	return iface, nil
}

// DeleteRouter deletes a router of a specified ID. A fatal error will occur
// if the deletion failed. This works best when used as a deferred function.
func DeleteRouter(t *testing.T, client *gophercloud.ServiceClient, routerID string) {
	t.Logf("Attempting to delete router: %s", routerID)

	err := routers.Delete(client, routerID).ExtractErr()
	if err != nil {
		t.Fatalf("Error deleting router: %v", err)
	}

	if err := WaitForRouterToDelete(client, routerID, 60); err != nil {
		t.Fatalf("Error waiting for router to delete: %v", err)
	}

	t.Logf("Deleted router: %s", routerID)
}

// DeleteRouterInterface will detach a subnet to a router. A fatal error will
// occur if the deletion failed. This works best when used as a deferred
// function.
func DeleteRouterInterface(t *testing.T, client *gophercloud.ServiceClient, portID, routerID string) {
	t.Logf("Attempting to detach port %s from router %s", portID, routerID)

	riOpts := routers.RemoveInterfaceOpts{
		PortID: portID,
	}

	_, err := routers.RemoveInterface(client, routerID, riOpts).Extract()
	if err != nil {
		t.Fatalf("Failed to detach port %s from router %s", portID, routerID)
	}

	if err := WaitForRouterInterfaceToDetach(client, portID, 60); err != nil {
		t.Fatalf("Failed to wait for port %s to detach from router %s", portID, routerID)
	}

	t.Logf("Successfully detached port %s from router %s", portID, routerID)
}

// DeleteFloatingIP deletes a floatingIP of a specified ID. A fatal error will
// occur if the deletion failed. This works best when used as a deferred
// function.
func DeleteFloatingIP(t *testing.T, client *gophercloud.ServiceClient, floatingIPID string) {
	t.Logf("Attempting to delete floating IP: %s", floatingIPID)

	err := floatingips.Delete(client, floatingIPID).ExtractErr()
	if err != nil {
		t.Fatalf("Failed to delete floating IP: %v", err)
	}

	t.Logf("Deleted floating IP: %s", floatingIPID)
}

// PrintFloatingIP prints a floating IP and all of its attributes.
func PrintFloatingIP(t *testing.T, fip *floatingips.FloatingIP) {
	t.Logf("ID: %s", fip.ID)
	t.Logf("FloatingNetworkID: %s", fip.FloatingNetworkID)
	t.Logf("FloatingIP: %s", fip.FloatingIP)
	t.Logf("PortID: %s", fip.PortID)
	t.Logf("FixedIP: %s", fip.FixedIP)
	t.Logf("TenantID: %s", fip.TenantID)
	t.Logf("Status: %s", fip.Status)
}

// PrintRouterInterface prints a router interface and all of its attributes.
func PrintRouterInterface(t *testing.T, routerInterface *routers.InterfaceInfo) {
	t.Logf("ID: %s", routerInterface.ID)
	t.Logf("SubnetID: %s", routerInterface.SubnetID)
	t.Logf("PortID: %s", routerInterface.PortID)
	t.Logf("TenantID: %s", routerInterface.TenantID)
}

// PrintRouter prints a router and all of its attributes.
func PrintRouter(t *testing.T, router *routers.Router) {
	t.Logf("ID: %s", router.ID)
	t.Logf("Status: %s", router.Status)
	t.Logf("GatewayInfo: %s", router.GatewayInfo)
	t.Logf("AdminStateUp: %t", router.AdminStateUp)
	t.Logf("Distributed: %t", router.Distributed)
	t.Logf("Name: %s", router.Name)
	t.Logf("TenantID: %s", router.TenantID)
	t.Logf("Routes:")

	for _, route := range router.Routes {
		t.Logf("\tNextHop: %s", route.NextHop)
		t.Logf("\tDestinationCIDR: %s", route.DestinationCIDR)
	}
}

func WaitForRouterToCreate(client *gophercloud.ServiceClient, routerID string, secs int) error {
	return gophercloud.WaitFor(secs, func() (bool, error) {
		r, err := routers.Get(client, routerID).Extract()
		if err != nil {
			return false, err
		}

		if r.Status == "ACTIVE" {
			return true, nil
		}

		return false, nil
	})
}

func WaitForRouterToDelete(client *gophercloud.ServiceClient, routerID string, secs int) error {
	return gophercloud.WaitFor(secs, func() (bool, error) {
		_, err := routers.Get(client, routerID).Extract()
		if err != nil {
			if _, ok := err.(gophercloud.ErrDefault404); ok {
				return true, nil
			}

			return false, err
		}

		return false, nil
	})
}

func WaitForRouterInterfaceToAttach(client *gophercloud.ServiceClient, routerInterfaceID string, secs int) error {
	return gophercloud.WaitFor(secs, func() (bool, error) {
		r, err := ports.Get(client, routerInterfaceID).Extract()
		if err != nil {
			return false, err
		}

		if r.Status == "ACTIVE" {
			return true, nil
		}

		return false, nil
	})
}

func WaitForRouterInterfaceToDetach(client *gophercloud.ServiceClient, routerInterfaceID string, secs int) error {
	return gophercloud.WaitFor(secs, func() (bool, error) {
		r, err := ports.Get(client, routerInterfaceID).Extract()
		if err != nil {
			if _, ok := err.(gophercloud.ErrDefault404); ok {
				return true, nil
			}

			if errCode, ok := err.(gophercloud.ErrUnexpectedResponseCode); ok {
				if errCode.Actual == 409 {
					return false, nil
				}
			}

			return false, err
		}

		if r.Status == "ACTIVE" {
			return true, nil
		}

		return false, nil
	})
}
