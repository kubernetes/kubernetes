package lbaas

import (
	"testing"

	base "github.com/rackspace/gophercloud/acceptance/openstack/networking/v2"
	"github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas/monitors"
	"github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas/pools"
	"github.com/rackspace/gophercloud/openstack/networking/v2/networks"
	"github.com/rackspace/gophercloud/openstack/networking/v2/subnets"
	th "github.com/rackspace/gophercloud/testhelper"
)

func SetupTopology(t *testing.T) (string, string) {
	// create network
	n, err := networks.Create(base.Client, networks.CreateOpts{Name: "tmp_network"}).Extract()
	th.AssertNoErr(t, err)

	t.Logf("Created network %s", n.ID)

	// create subnet
	s, err := subnets.Create(base.Client, subnets.CreateOpts{
		NetworkID: n.ID,
		CIDR:      "192.168.199.0/24",
		IPVersion: subnets.IPv4,
		Name:      "tmp_subnet",
	}).Extract()
	th.AssertNoErr(t, err)

	t.Logf("Created subnet %s", s.ID)

	return n.ID, s.ID
}

func DeleteTopology(t *testing.T, networkID string) {
	res := networks.Delete(base.Client, networkID)
	th.AssertNoErr(t, res.Err)
	t.Logf("Deleted network %s", networkID)
}

func CreatePool(t *testing.T, subnetID string) string {
	p, err := pools.Create(base.Client, pools.CreateOpts{
		LBMethod: pools.LBMethodRoundRobin,
		Protocol: "HTTP",
		Name:     "tmp_pool",
		SubnetID: subnetID,
	}).Extract()

	th.AssertNoErr(t, err)

	t.Logf("Created pool %s", p.ID)

	return p.ID
}

func DeletePool(t *testing.T, poolID string) {
	res := pools.Delete(base.Client, poolID)
	th.AssertNoErr(t, res.Err)
	t.Logf("Deleted pool %s", poolID)
}

func CreateMonitor(t *testing.T) string {
	m, err := monitors.Create(base.Client, monitors.CreateOpts{
		Delay:         10,
		Timeout:       10,
		MaxRetries:    3,
		Type:          monitors.TypeHTTP,
		ExpectedCodes: "200",
		URLPath:       "/login",
		HTTPMethod:    "GET",
	}).Extract()

	th.AssertNoErr(t, err)

	t.Logf("Created monitor ID [%s]", m.ID)

	return m.ID
}
