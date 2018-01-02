// +build acceptance networking lbaas pool

package lbaas

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	networking "github.com/gophercloud/gophercloud/acceptance/openstack/networking/v2"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/lbaas/pools"
)

func TestPoolsList(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	if err != nil {
		t.Fatalf("Unable to create a network client: %v", err)
	}

	allPages, err := pools.List(client, pools.ListOpts{}).AllPages()
	if err != nil {
		t.Fatalf("Unable to list pools: %v", err)
	}

	allPools, err := pools.ExtractPools(allPages)
	if err != nil {
		t.Fatalf("Unable to extract pools: %v", err)
	}

	for _, pool := range allPools {
		tools.PrintResource(t, pool)
	}
}

func TestPoolsCRUD(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	if err != nil {
		t.Fatalf("Unable to create a network client: %v", err)
	}

	network, err := networking.CreateNetwork(t, client)
	if err != nil {
		t.Fatalf("Unable to create network: %v", err)
	}
	defer networking.DeleteNetwork(t, client, network.ID)

	subnet, err := networking.CreateSubnet(t, client, network.ID)
	if err != nil {
		t.Fatalf("Unable to create subnet: %v", err)
	}
	defer networking.DeleteSubnet(t, client, subnet.ID)

	pool, err := CreatePool(t, client, subnet.ID)
	if err != nil {
		t.Fatalf("Unable to create pool: %v", err)
	}
	defer DeletePool(t, client, pool.ID)

	tools.PrintResource(t, pool)

	updateOpts := pools.UpdateOpts{
		LBMethod: pools.LBMethodLeastConnections,
	}

	_, err = pools.Update(client, pool.ID, updateOpts).Extract()
	if err != nil {
		t.Fatalf("Unable to update pool: %v")
	}

	newPool, err := pools.Get(client, pool.ID).Extract()
	if err != nil {
		t.Fatalf("Unable to get pool: %v")
	}

	tools.PrintResource(t, newPool)
}

func TestPoolsMonitors(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	if err != nil {
		t.Fatalf("Unable to create a network client: %v", err)
	}

	network, err := networking.CreateNetwork(t, client)
	if err != nil {
		t.Fatalf("Unable to create network: %v", err)
	}
	defer networking.DeleteNetwork(t, client, network.ID)

	subnet, err := networking.CreateSubnet(t, client, network.ID)
	if err != nil {
		t.Fatalf("Unable to create subnet: %v", err)
	}
	defer networking.DeleteSubnet(t, client, subnet.ID)

	pool, err := CreatePool(t, client, subnet.ID)
	if err != nil {
		t.Fatalf("Unable to create pool: %v", err)
	}
	defer DeletePool(t, client, pool.ID)

	monitor, err := CreateMonitor(t, client)
	if err != nil {
		t.Fatalf("Unable to create monitor: %v", err)
	}
	defer DeleteMonitor(t, client, monitor.ID)

	t.Logf("Associating monitor %s with pool %s", monitor.ID, pool.ID)
	if res := pools.AssociateMonitor(client, pool.ID, monitor.ID); res.Err != nil {
		t.Fatalf("Unable to associate monitor to pool")
	}

	t.Logf("Disassociating monitor %s with pool %s", monitor.ID, pool.ID)
	if res := pools.DisassociateMonitor(client, pool.ID, monitor.ID); res.Err != nil {
		t.Fatalf("Unable to disassociate monitor from pool")
	}

}
