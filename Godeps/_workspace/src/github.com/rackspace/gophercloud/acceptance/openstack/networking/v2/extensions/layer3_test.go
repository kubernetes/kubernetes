// +build acceptance networking layer3ext

package extensions

import (
	"testing"

	base "github.com/rackspace/gophercloud/acceptance/openstack/networking/v2"
	"github.com/rackspace/gophercloud/openstack/networking/v2/extensions/external"
	"github.com/rackspace/gophercloud/openstack/networking/v2/extensions/layer3/floatingips"
	"github.com/rackspace/gophercloud/openstack/networking/v2/extensions/layer3/routers"
	"github.com/rackspace/gophercloud/openstack/networking/v2/networks"
	"github.com/rackspace/gophercloud/openstack/networking/v2/ports"
	"github.com/rackspace/gophercloud/openstack/networking/v2/subnets"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
)

const (
	cidr1 = "10.0.0.1/24"
	cidr2 = "20.0.0.1/24"
)

func TestAll(t *testing.T) {
	base.Setup(t)
	defer base.Teardown()

	testRouter(t)
	testFloatingIP(t)
}

func testRouter(t *testing.T) {
	// Setup: Create network
	networkID := createNetwork(t)

	// Create router
	routerID := createRouter(t, networkID)

	// Lists routers
	listRouters(t)

	// Update router
	updateRouter(t, routerID)

	// Get router
	getRouter(t, routerID)

	// Create new subnet. Note: this subnet will be deleted when networkID is deleted
	subnetID := createSubnet(t, networkID, cidr2)

	// Add interface
	addInterface(t, routerID, subnetID)

	// Remove interface
	removeInterface(t, routerID, subnetID)

	// Delete router
	deleteRouter(t, routerID)

	// Cleanup
	deleteNetwork(t, networkID)
}

func testFloatingIP(t *testing.T) {
	// Setup external network
	extNetworkID := createNetwork(t)

	// Setup internal network, subnet and port
	intNetworkID, subnetID, portID := createInternalTopology(t)

	// Now the important part: we need to allow the external network to talk to
	// the internal subnet. For this we need a router that has an interface to
	// the internal subnet.
	routerID := bridgeIntSubnetWithExtNetwork(t, extNetworkID, subnetID)

	// Create floating IP
	ipID := createFloatingIP(t, extNetworkID, portID)

	// Get floating IP
	getFloatingIP(t, ipID)

	// Update floating IP
	updateFloatingIP(t, ipID, portID)

	// Delete floating IP
	deleteFloatingIP(t, ipID)

	// Remove the internal subnet interface
	removeInterface(t, routerID, subnetID)

	// Delete router and external network
	deleteRouter(t, routerID)
	deleteNetwork(t, extNetworkID)

	// Delete internal port and network
	deletePort(t, portID)
	deleteNetwork(t, intNetworkID)
}

func createNetwork(t *testing.T) string {
	t.Logf("Creating a network")

	asu := true
	opts := external.CreateOpts{
		Parent:   networks.CreateOpts{Name: "sample_network", AdminStateUp: &asu},
		External: true,
	}
	n, err := networks.Create(base.Client, opts).Extract()

	th.AssertNoErr(t, err)

	if n.ID == "" {
		t.Fatalf("No ID returned when creating a network")
	}

	createSubnet(t, n.ID, cidr1)

	t.Logf("Network created: ID [%s]", n.ID)

	return n.ID
}

func deleteNetwork(t *testing.T, networkID string) {
	t.Logf("Deleting network %s", networkID)
	networks.Delete(base.Client, networkID)
}

func deletePort(t *testing.T, portID string) {
	t.Logf("Deleting port %s", portID)
	ports.Delete(base.Client, portID)
}

func createInternalTopology(t *testing.T) (string, string, string) {
	t.Logf("Creating an internal network (for port)")
	opts := networks.CreateOpts{Name: "internal_network"}
	n, err := networks.Create(base.Client, opts).Extract()
	th.AssertNoErr(t, err)

	// A subnet is also needed
	subnetID := createSubnet(t, n.ID, cidr2)

	t.Logf("Creating an internal port on network %s", n.ID)
	p, err := ports.Create(base.Client, ports.CreateOpts{
		NetworkID: n.ID,
		Name:      "fixed_internal_port",
	}).Extract()
	th.AssertNoErr(t, err)

	return n.ID, subnetID, p.ID
}

func bridgeIntSubnetWithExtNetwork(t *testing.T, networkID, subnetID string) string {
	// Create router with external gateway info
	routerID := createRouter(t, networkID)

	// Add interface for internal subnet
	addInterface(t, routerID, subnetID)

	return routerID
}

func createSubnet(t *testing.T, networkID, cidr string) string {
	t.Logf("Creating a subnet for network %s", networkID)

	iFalse := false
	s, err := subnets.Create(base.Client, subnets.CreateOpts{
		NetworkID:  networkID,
		CIDR:       cidr,
		IPVersion:  subnets.IPv4,
		Name:       "my_subnet",
		EnableDHCP: &iFalse,
	}).Extract()

	th.AssertNoErr(t, err)

	t.Logf("Subnet created: ID [%s]", s.ID)

	return s.ID
}

func createRouter(t *testing.T, networkID string) string {
	t.Logf("Creating a router for network %s", networkID)

	asu := false
	gwi := routers.GatewayInfo{NetworkID: networkID}
	r, err := routers.Create(base.Client, routers.CreateOpts{
		Name:         "foo_router",
		AdminStateUp: &asu,
		GatewayInfo:  &gwi,
	}).Extract()

	th.AssertNoErr(t, err)

	if r.ID == "" {
		t.Fatalf("No ID returned when creating a router")
	}

	t.Logf("Router created: ID [%s]", r.ID)

	return r.ID
}

func listRouters(t *testing.T) {
	pager := routers.List(base.Client, routers.ListOpts{})

	err := pager.EachPage(func(page pagination.Page) (bool, error) {
		routerList, err := routers.ExtractRouters(page)
		th.AssertNoErr(t, err)

		for _, r := range routerList {
			t.Logf("Listing router: ID [%s] Name [%s] Status [%s] GatewayInfo [%#v]",
				r.ID, r.Name, r.Status, r.GatewayInfo)
		}

		return true, nil
	})

	th.AssertNoErr(t, err)
}

func updateRouter(t *testing.T, routerID string) {
	_, err := routers.Update(base.Client, routerID, routers.UpdateOpts{
		Name: "another_name",
	}).Extract()

	th.AssertNoErr(t, err)
}

func getRouter(t *testing.T, routerID string) {
	r, err := routers.Get(base.Client, routerID).Extract()

	th.AssertNoErr(t, err)

	t.Logf("Getting router: ID [%s] Name [%s] Status [%s]", r.ID, r.Name, r.Status)
}

func addInterface(t *testing.T, routerID, subnetID string) {
	ir, err := routers.AddInterface(base.Client, routerID, routers.InterfaceOpts{SubnetID: subnetID}).Extract()

	th.AssertNoErr(t, err)

	t.Logf("Interface added to router %s: SubnetID [%s] PortID [%s]", routerID, ir.SubnetID, ir.PortID)
}

func removeInterface(t *testing.T, routerID, subnetID string) {
	ir, err := routers.RemoveInterface(base.Client, routerID, routers.InterfaceOpts{SubnetID: subnetID}).Extract()

	th.AssertNoErr(t, err)

	t.Logf("Interface %s removed from %s", ir.ID, routerID)
}

func deleteRouter(t *testing.T, routerID string) {
	t.Logf("Deleting router %s", routerID)

	res := routers.Delete(base.Client, routerID)

	th.AssertNoErr(t, res.Err)
}

func createFloatingIP(t *testing.T, networkID, portID string) string {
	t.Logf("Creating floating IP on network [%s] with port [%s]", networkID, portID)

	opts := floatingips.CreateOpts{
		FloatingNetworkID: networkID,
		PortID:            portID,
	}

	ip, err := floatingips.Create(base.Client, opts).Extract()

	th.AssertNoErr(t, err)

	t.Logf("Floating IP created: ID [%s] Status [%s] Fixed (internal) IP: [%s] Floating (external) IP: [%s]",
		ip.ID, ip.Status, ip.FixedIP, ip.FloatingIP)

	return ip.ID
}

func getFloatingIP(t *testing.T, ipID string) {
	ip, err := floatingips.Get(base.Client, ipID).Extract()
	th.AssertNoErr(t, err)

	t.Logf("Getting floating IP: ID [%s] Status [%s]", ip.ID, ip.Status)
}

func updateFloatingIP(t *testing.T, ipID, portID string) {
	t.Logf("Disassociate all ports from IP %s", ipID)
	_, err := floatingips.Update(base.Client, ipID, floatingips.UpdateOpts{PortID: ""}).Extract()
	th.AssertNoErr(t, err)

	t.Logf("Re-associate the port %s", portID)
	_, err = floatingips.Update(base.Client, ipID, floatingips.UpdateOpts{PortID: portID}).Extract()
	th.AssertNoErr(t, err)
}

func deleteFloatingIP(t *testing.T, ipID string) {
	t.Logf("Deleting IP %s", ipID)
	res := floatingips.Delete(base.Client, ipID)
	th.AssertNoErr(t, res.Err)
}
