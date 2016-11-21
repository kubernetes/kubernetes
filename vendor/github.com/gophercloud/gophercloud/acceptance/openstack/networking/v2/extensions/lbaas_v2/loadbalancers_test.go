// +build acceptance networking lbaas_v2 loadbalancers

package lbaas_v2

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	networking "github.com/gophercloud/gophercloud/acceptance/openstack/networking/v2"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/lbaas_v2/listeners"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/lbaas_v2/loadbalancers"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/lbaas_v2/monitors"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/lbaas_v2/pools"
)

func TestLoadbalancersList(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	if err != nil {
		t.Fatalf("Unable to create a network client: %v", err)
	}

	allPages, err := loadbalancers.List(client, nil).AllPages()
	if err != nil {
		t.Fatalf("Unable to list loadbalancers: %v", err)
	}

	allLoadbalancers, err := loadbalancers.ExtractLoadBalancers(allPages)
	if err != nil {
		t.Fatalf("Unable to extract loadbalancers: %v", err)
	}

	for _, lb := range allLoadbalancers {
		PrintLoadBalancer(t, &lb)
	}
}

func TestLoadbalancersCRUD(t *testing.T) {
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

	lb, err := CreateLoadBalancer(t, client, subnet.ID)
	if err != nil {
		t.Fatalf("Unable to create loadbalancer: %v", err)
	}
	defer DeleteLoadBalancer(t, client, lb.ID)

	newLB, err := loadbalancers.Get(client, lb.ID).Extract()
	if err != nil {
		t.Fatalf("Unable to get loadbalancer: %v", err)
	}

	PrintLoadBalancer(t, newLB)

	// Because of the time it takes to create a loadbalancer,
	// this test will include some other resources.

	// Listener
	listener, err := CreateListener(t, client, lb)
	if err != nil {
		t.Fatalf("Unable to create listener: %v", err)
	}
	defer DeleteListener(t, client, lb.ID, listener.ID)

	updateListenerOpts := listeners.UpdateOpts{
		Description: "Some listener description",
	}
	_, err = listeners.Update(client, listener.ID, updateListenerOpts).Extract()
	if err != nil {
		t.Fatalf("Unable to update listener")
	}

	if err := WaitForLoadBalancerState(client, lb.ID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		t.Fatalf("Timed out waiting for loadbalancer to become active")
	}

	newListener, err := listeners.Get(client, listener.ID).Extract()
	if err != nil {
		t.Fatalf("Unable to get listener")
	}

	PrintListener(t, newListener)

	// Pool
	pool, err := CreatePool(t, client, lb)
	if err != nil {
		t.Fatalf("Unable to create pool: %v", err)
	}
	defer DeletePool(t, client, lb.ID, pool.ID)

	updatePoolOpts := pools.UpdateOpts{
		Description: "Some pool description",
	}
	_, err = pools.Update(client, pool.ID, updatePoolOpts).Extract()
	if err != nil {
		t.Fatalf("Unable to update pool")
	}

	if err := WaitForLoadBalancerState(client, lb.ID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		t.Fatalf("Timed out waiting for loadbalancer to become active")
	}

	newPool, err := pools.Get(client, pool.ID).Extract()
	if err != nil {
		t.Fatalf("Unable to get pool")
	}

	PrintPool(t, newPool)

	// Member
	member, err := CreateMember(t, client, lb, newPool, subnet.ID, subnet.CIDR)
	if err != nil {
		t.Fatalf("Unable to create member: %v", err)
	}
	defer DeleteMember(t, client, lb.ID, pool.ID, member.ID)

	newWeight := tools.RandomInt(11, 100)
	updateMemberOpts := pools.UpdateMemberOpts{
		Weight: newWeight,
	}
	_, err = pools.UpdateMember(client, pool.ID, member.ID, updateMemberOpts).Extract()
	if err != nil {
		t.Fatalf("Unable to update pool")
	}

	if err := WaitForLoadBalancerState(client, lb.ID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		t.Fatalf("Timed out waiting for loadbalancer to become active")
	}

	newMember, err := pools.GetMember(client, pool.ID, member.ID).Extract()
	if err != nil {
		t.Fatalf("Unable to get member")
	}

	PrintMember(t, newMember)

	// Monitor
	monitor, err := CreateMonitor(t, client, lb, newPool)
	if err != nil {
		t.Fatalf("Unable to create monitor: %v", err)
	}
	defer DeleteMonitor(t, client, lb.ID, monitor.ID)

	newDelay := tools.RandomInt(20, 30)
	updateMonitorOpts := monitors.UpdateOpts{
		Delay: newDelay,
	}
	_, err = monitors.Update(client, monitor.ID, updateMonitorOpts).Extract()
	if err != nil {
		t.Fatalf("Unable to update monitor")
	}

	if err := WaitForLoadBalancerState(client, lb.ID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		t.Fatalf("Timed out waiting for loadbalancer to become active")
	}

	newMonitor, err := monitors.Get(client, monitor.ID).Extract()
	if err != nil {
		t.Fatalf("Unable to get monitor")
	}

	PrintMonitor(t, newMonitor)

}
