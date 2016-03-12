// +build acceptance networking lbaas lbaaspool

package lbaas

import (
	"testing"

	base "github.com/rackspace/gophercloud/acceptance/openstack/networking/v2"
	"github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas/pools"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestPools(t *testing.T) {
	base.Setup(t)
	defer base.Teardown()

	// setup
	networkID, subnetID := SetupTopology(t)

	// create pool
	poolID := CreatePool(t, subnetID)

	// list pools
	listPools(t)

	// update pool
	updatePool(t, poolID)

	// get pool
	getPool(t, poolID)

	// create monitor
	monitorID := CreateMonitor(t)

	// associate health monitor
	associateMonitor(t, poolID, monitorID)

	// disassociate health monitor
	disassociateMonitor(t, poolID, monitorID)

	// delete pool
	DeletePool(t, poolID)

	// teardown
	DeleteTopology(t, networkID)
}

func listPools(t *testing.T) {
	err := pools.List(base.Client, pools.ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		poolList, err := pools.ExtractPools(page)
		if err != nil {
			t.Errorf("Failed to extract pools: %v", err)
			return false, err
		}

		for _, p := range poolList {
			t.Logf("Listing pool: ID [%s] Name [%s] Status [%s] LB algorithm [%s]", p.ID, p.Name, p.Status, p.LBMethod)
		}

		return true, nil
	})

	th.AssertNoErr(t, err)
}

func updatePool(t *testing.T, poolID string) {
	opts := pools.UpdateOpts{Name: "SuperPool", LBMethod: pools.LBMethodLeastConnections}
	p, err := pools.Update(base.Client, poolID, opts).Extract()

	th.AssertNoErr(t, err)

	t.Logf("Updated pool ID [%s]", p.ID)
}

func getPool(t *testing.T, poolID string) {
	p, err := pools.Get(base.Client, poolID).Extract()

	th.AssertNoErr(t, err)

	t.Logf("Getting pool ID [%s]", p.ID)
}

func associateMonitor(t *testing.T, poolID, monitorID string) {
	res := pools.AssociateMonitor(base.Client, poolID, monitorID)

	th.AssertNoErr(t, res.Err)

	t.Logf("Associated pool %s with monitor %s", poolID, monitorID)
}

func disassociateMonitor(t *testing.T, poolID, monitorID string) {
	res := pools.DisassociateMonitor(base.Client, poolID, monitorID)

	th.AssertNoErr(t, res.Err)

	t.Logf("Disassociated pool %s with monitor %s", poolID, monitorID)
}
