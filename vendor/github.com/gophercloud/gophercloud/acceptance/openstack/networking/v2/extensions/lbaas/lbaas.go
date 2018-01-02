package lbaas

import (
	"fmt"
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/lbaas/members"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/lbaas/monitors"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/lbaas/pools"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/lbaas/vips"
)

// CreateMember will create a load balancer member in a specified pool on a
// random port. An error will be returned if the member could not be created.
func CreateMember(t *testing.T, client *gophercloud.ServiceClient, poolID string) (*members.Member, error) {
	protocolPort := tools.RandomInt(100, 1000)
	address := tools.RandomInt(2, 200)
	t.Logf("Attempting to create member in port %d", protocolPort)

	createOpts := members.CreateOpts{
		PoolID:       poolID,
		ProtocolPort: protocolPort,
		Address:      fmt.Sprintf("192.168.1.%d", address),
	}

	member, err := members.Create(client, createOpts).Extract()
	if err != nil {
		return member, err
	}

	t.Logf("Successfully created member %s", member.ID)

	return member, nil
}

// CreateMonitor will create a monitor with a random name for a specific pool.
// An error will be returned if the monitor could not be created.
func CreateMonitor(t *testing.T, client *gophercloud.ServiceClient) (*monitors.Monitor, error) {
	t.Logf("Attempting to create monitor.")

	createOpts := monitors.CreateOpts{
		Type:         monitors.TypePING,
		Delay:        90,
		Timeout:      60,
		MaxRetries:   10,
		AdminStateUp: gophercloud.Enabled,
	}

	monitor, err := monitors.Create(client, createOpts).Extract()
	if err != nil {
		return monitor, err
	}

	t.Logf("Successfully created monitor %s", monitor.ID)

	return monitor, nil
}

// CreatePool will create a pool with a random name. An error will be returned
// if the pool could not be deleted.
func CreatePool(t *testing.T, client *gophercloud.ServiceClient, subnetID string) (*pools.Pool, error) {
	poolName := tools.RandomString("TESTACCT-", 8)

	t.Logf("Attempting to create pool %s", poolName)

	createOpts := pools.CreateOpts{
		Name:     poolName,
		SubnetID: subnetID,
		Protocol: pools.ProtocolTCP,
		LBMethod: pools.LBMethodRoundRobin,
	}

	pool, err := pools.Create(client, createOpts).Extract()
	if err != nil {
		return pool, err
	}

	t.Logf("Successfully created pool %s", poolName)

	return pool, nil
}

// CreateVIP will create a vip with a random name and a random port in a
// specified subnet and pool. An error will be returned if the vip could
// not be created.
func CreateVIP(t *testing.T, client *gophercloud.ServiceClient, subnetID, poolID string) (*vips.VirtualIP, error) {
	vipName := tools.RandomString("TESTACCT-", 8)
	vipPort := tools.RandomInt(100, 10000)

	t.Logf("Attempting to create VIP %s", vipName)

	createOpts := vips.CreateOpts{
		Name:         vipName,
		SubnetID:     subnetID,
		PoolID:       poolID,
		Protocol:     "TCP",
		ProtocolPort: vipPort,
	}

	vip, err := vips.Create(client, createOpts).Extract()
	if err != nil {
		return vip, err
	}

	t.Logf("Successfully created vip %s", vipName)

	return vip, nil
}

// DeleteMember will delete a specified member. A fatal error will occur if
// the member could not be deleted. This works best when used as a deferred
// function.
func DeleteMember(t *testing.T, client *gophercloud.ServiceClient, memberID string) {
	t.Logf("Attempting to delete member %s", memberID)

	if err := members.Delete(client, memberID).ExtractErr(); err != nil {
		t.Fatalf("Unable to delete member: %v", err)
	}

	t.Logf("Successfully deleted member %s", memberID)
}

// DeleteMonitor will delete a specified monitor. A fatal error will occur if
// the monitor could not be deleted. This works best when used as a deferred
// function.
func DeleteMonitor(t *testing.T, client *gophercloud.ServiceClient, monitorID string) {
	t.Logf("Attempting to delete monitor %s", monitorID)

	if err := monitors.Delete(client, monitorID).ExtractErr(); err != nil {
		t.Fatalf("Unable to delete monitor: %v", err)
	}

	t.Logf("Successfully deleted monitor %s", monitorID)
}

// DeletePool will delete a specified pool. A fatal error will occur if the
// pool could not be deleted. This works best when used as a deferred function.
func DeletePool(t *testing.T, client *gophercloud.ServiceClient, poolID string) {
	t.Logf("Attempting to delete pool %s", poolID)

	if err := pools.Delete(client, poolID).ExtractErr(); err != nil {
		t.Fatalf("Unable to delete pool: %v", err)
	}

	t.Logf("Successfully deleted pool %s", poolID)
}

// DeleteVIP will delete a specified vip. A fatal error will occur if the vip
// could not be deleted. This works best when used as a deferred function.
func DeleteVIP(t *testing.T, client *gophercloud.ServiceClient, vipID string) {
	t.Logf("Attempting to delete vip %s", vipID)

	if err := vips.Delete(client, vipID).ExtractErr(); err != nil {
		t.Fatalf("Unable to delete vip: %v", err)
	}

	t.Logf("Successfully deleted vip %s", vipID)
}
