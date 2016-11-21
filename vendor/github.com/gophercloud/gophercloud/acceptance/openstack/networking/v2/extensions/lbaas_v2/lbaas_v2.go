package lbaas_v2

import (
	"fmt"
	"strings"
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/lbaas_v2/listeners"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/lbaas_v2/loadbalancers"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/lbaas_v2/monitors"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/lbaas_v2/pools"
)

const loadbalancerActiveTimeoutSeconds = 300
const loadbalancerDeleteTimeoutSeconds = 300

// CreateListener will create a listener for a given load balancer on a random
// port with a random name. An error will be returned if the listener could not
// be created.
func CreateListener(t *testing.T, client *gophercloud.ServiceClient, lb *loadbalancers.LoadBalancer) (*listeners.Listener, error) {
	listenerName := tools.RandomString("TESTACCT-", 8)
	listenerPort := tools.RandomInt(1, 100)

	t.Logf("Attempting to create listener %s on port %d", listenerName, listenerPort)

	createOpts := listeners.CreateOpts{
		Name:           listenerName,
		LoadbalancerID: lb.ID,
		Protocol:       "TCP",
		ProtocolPort:   listenerPort,
	}

	listener, err := listeners.Create(client, createOpts).Extract()
	if err != nil {
		return listener, err
	}

	t.Logf("Successfully created listener %s", listenerName)

	if err := WaitForLoadBalancerState(client, lb.ID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		return listener, fmt.Errorf("Timed out waiting for loadbalancer to become active")
	}

	return listener, nil
}

// CreateLoadBalancer will create a load balancer with a random name on a given
// subnet. An error will be returned if the loadbalancer could not be created.
func CreateLoadBalancer(t *testing.T, client *gophercloud.ServiceClient, subnetID string) (*loadbalancers.LoadBalancer, error) {
	lbName := tools.RandomString("TESTACCT-", 8)

	t.Logf("Attempting to create loadbalancer %s on subnet %s", lbName, subnetID)

	createOpts := loadbalancers.CreateOpts{
		Name:         lbName,
		VipSubnetID:  subnetID,
		AdminStateUp: gophercloud.Enabled,
	}

	lb, err := loadbalancers.Create(client, createOpts).Extract()
	if err != nil {
		return lb, err
	}

	t.Logf("Successfully created loadbalancer %s on subnet %s", lbName, subnetID)
	t.Logf("Waiting for loadbalancer %s to become active", lbName)

	if err := WaitForLoadBalancerState(client, lb.ID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		return lb, err
	}

	t.Logf("LoadBalancer %s is active", lbName)

	return lb, nil
}

// CreateMember will create a member with a random name, port, address, and
// weight. An error will be returned if the member could not be created.
func CreateMember(t *testing.T, client *gophercloud.ServiceClient, lb *loadbalancers.LoadBalancer, pool *pools.Pool, subnetID, subnetCIDR string) (*pools.Member, error) {
	memberName := tools.RandomString("TESTACCT-", 8)
	memberPort := tools.RandomInt(100, 1000)
	memberWeight := tools.RandomInt(1, 10)

	cidrParts := strings.Split(subnetCIDR, "/")
	subnetParts := strings.Split(cidrParts[0], ".")
	memberAddress := fmt.Sprintf("%s.%s.%s.%d", subnetParts[0], subnetParts[1], subnetParts[2], tools.RandomInt(10, 100))

	t.Logf("Attempting to create member %s", memberName)

	createOpts := pools.CreateMemberOpts{
		Name:         memberName,
		ProtocolPort: memberPort,
		Weight:       memberWeight,
		Address:      memberAddress,
		SubnetID:     subnetID,
	}

	t.Logf("Member create opts: %#v", createOpts)

	member, err := pools.CreateMember(client, pool.ID, createOpts).Extract()
	if err != nil {
		return member, err
	}

	t.Logf("Successfully created member %s", memberName)

	if err := WaitForLoadBalancerState(client, lb.ID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		return member, fmt.Errorf("Timed out waiting for loadbalancer to become active")
	}

	return member, nil
}

// CreateMonitor will create a monitor with a random name for a specific pool.
// An error will be returned if the monitor could not be created.
func CreateMonitor(t *testing.T, client *gophercloud.ServiceClient, lb *loadbalancers.LoadBalancer, pool *pools.Pool) (*monitors.Monitor, error) {
	monitorName := tools.RandomString("TESTACCT-", 8)

	t.Logf("Attempting to create monitor %s", monitorName)

	createOpts := monitors.CreateOpts{
		PoolID:     pool.ID,
		Name:       monitorName,
		Delay:      10,
		Timeout:    5,
		MaxRetries: 5,
		Type:       "PING",
	}

	monitor, err := monitors.Create(client, createOpts).Extract()
	if err != nil {
		return monitor, err
	}

	t.Logf("Successfully created monitor: %s", monitorName)

	if err := WaitForLoadBalancerState(client, lb.ID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		return monitor, fmt.Errorf("Timed out waiting for loadbalancer to become active")
	}

	return monitor, nil
}

// CreatePool will create a pool with a random name with a specified listener
// and loadbalancer. An error will be returned if the pool could not be
// created.
func CreatePool(t *testing.T, client *gophercloud.ServiceClient, lb *loadbalancers.LoadBalancer) (*pools.Pool, error) {
	poolName := tools.RandomString("TESTACCT-", 8)

	t.Logf("Attempting to create pool %s", poolName)

	createOpts := pools.CreateOpts{
		Name:           poolName,
		Protocol:       pools.ProtocolTCP,
		LoadbalancerID: lb.ID,
		LBMethod:       pools.LBMethodLeastConnections,
	}

	pool, err := pools.Create(client, createOpts).Extract()
	if err != nil {
		return pool, err
	}

	t.Logf("Successfully created pool %s", poolName)

	if err := WaitForLoadBalancerState(client, lb.ID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		return pool, fmt.Errorf("Timed out waiting for loadbalancer to become active")
	}

	return pool, nil
}

// DeleteListener will delete a specified listener. A fatal error will occur if
// the listener could not be deleted. This works best when used as a deferred
// function.
func DeleteListener(t *testing.T, client *gophercloud.ServiceClient, lbID, listenerID string) {
	t.Logf("Attempting to delete listener %s", listenerID)

	if err := listeners.Delete(client, listenerID).ExtractErr(); err != nil {
		t.Fatalf("Unable to delete listener: %v", err)
	}

	if err := WaitForLoadBalancerState(client, lbID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		t.Fatalf("Timed out waiting for loadbalancer to become active")
	}

	t.Logf("Successfully deleted listener %s", listenerID)
}

// DeleteMember will delete a specified member. A fatal error will occur if the
// member could not be deleted. This works best when used as a deferred
// function.
func DeleteMember(t *testing.T, client *gophercloud.ServiceClient, lbID, poolID, memberID string) {
	t.Logf("Attempting to delete member %s", memberID)

	if err := pools.DeleteMember(client, poolID, memberID).ExtractErr(); err != nil {
		t.Fatalf("Unable to delete member: %s", memberID)
	}

	if err := WaitForLoadBalancerState(client, lbID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		t.Fatalf("Timed out waiting for loadbalancer to become active")
	}

	t.Logf("Successfully deleted member %s", memberID)
}

// DeleteLoadBalancer will delete a specified loadbalancer. A fatal error will
// occur if the loadbalancer could not be deleted. This works best when used
// as a deferred function.
func DeleteLoadBalancer(t *testing.T, client *gophercloud.ServiceClient, lbID string) {
	t.Logf("Attempting to delete loadbalancer %s", lbID)

	if err := loadbalancers.Delete(client, lbID).ExtractErr(); err != nil {
		t.Fatalf("Unable to delete loadbalancer: %v", err)
	}

	t.Logf("Waiting for loadbalancer %s to delete", lbID)

	if err := WaitForLoadBalancerState(client, lbID, "DELETED", loadbalancerActiveTimeoutSeconds); err != nil {
		t.Fatalf("Loadbalancer did not delete in time.")
	}

	t.Logf("Successfully deleted loadbalancer %s", lbID)
}

// DeleteMonitor will delete a specified monitor. A fatal error will occur if
// the monitor could not be deleted. This works best when used as a deferred
// function.
func DeleteMonitor(t *testing.T, client *gophercloud.ServiceClient, lbID, monitorID string) {
	t.Logf("Attempting to delete monitor %s", monitorID)

	if err := monitors.Delete(client, monitorID).ExtractErr(); err != nil {
		t.Fatalf("Unable to delete monitor: %v", err)
	}

	if err := WaitForLoadBalancerState(client, lbID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		t.Fatalf("Timed out waiting for loadbalancer to become active")
	}

	t.Logf("Successfully deleted monitor %s", monitorID)
}

// DeletePool will delete a specified pool. A fatal error will occur if the
// pool could not be deleted. This works best when used as a deferred function.
func DeletePool(t *testing.T, client *gophercloud.ServiceClient, lbID, poolID string) {
	t.Logf("Attempting to delete pool %s", poolID)

	if err := pools.Delete(client, poolID).ExtractErr(); err != nil {
		t.Fatalf("Unable to delete pool: %v", err)
	}

	if err := WaitForLoadBalancerState(client, lbID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		t.Fatalf("Timed out waiting for loadbalancer to become active")
	}

	t.Logf("Successfully deleted pool %s", poolID)
}

// PrintListener will print a listener and all of its attributes.
func PrintListener(t *testing.T, listener *listeners.Listener) {
	t.Logf("ID: %s", listener.ID)
	t.Logf("TenantID: %s", listener.TenantID)
	t.Logf("Name: %s", listener.Name)
	t.Logf("Description: %s", listener.Description)
	t.Logf("Protocol: %s", listener.Protocol)
	t.Logf("DefaultPoolID: %s", listener.DefaultPoolID)
	t.Logf("ConnLimit: %d", listener.ConnLimit)
	t.Logf("SniContainerRefs: %s", listener.SniContainerRefs)
	t.Logf("DefaultTlsContainerRef: %s", listener.DefaultTlsContainerRef)
	t.Logf("AdminStateUp: %t", listener.AdminStateUp)

	t.Logf("Pools:")
	for _, pool := range listener.Pools {
		t.Logf("\t%#v", pool)
	}

	t.Logf("LoadBalancers")
	for _, lb := range listener.Loadbalancers {
		t.Logf("\t%#v", lb)
	}
}

// PrintLoadBalancer will print a load balancer and all of its attributes.
func PrintLoadBalancer(t *testing.T, lb *loadbalancers.LoadBalancer) {
	t.Logf("ID: %s", lb.ID)
	t.Logf("Name: %s", lb.Name)
	t.Logf("TenantID: %s", lb.TenantID)
	t.Logf("Description: %s", lb.Description)
	t.Logf("ProvisioningStatus: %s", lb.ProvisioningStatus)
	t.Logf("VipAddress: %s", lb.VipAddress)
	t.Logf("VipPortID: %s", lb.VipPortID)
	t.Logf("VipSubnetID: %s", lb.VipSubnetID)
	t.Logf("OperatingStatus: %s", lb.OperatingStatus)
	t.Logf("Flavor: %s", lb.Flavor)
	t.Logf("Provider: %s", lb.Provider)
	t.Logf("AdminStateUp: %t", lb.AdminStateUp)

	t.Logf("Listeners")
	for _, listener := range lb.Listeners {
		t.Logf("\t%#v", listener)
	}
}

// PrintMember will print a member and all of its attributes.
func PrintMember(t *testing.T, member *pools.Member) {
	t.Logf("ID: %s", member.ID)
	t.Logf("Name: %s", member.Name)
	t.Logf("TenantID: %s", member.TenantID)
	t.Logf("Weight: %d", member.Weight)
	t.Logf("SubnetID: %s", member.SubnetID)
	t.Logf("PoolID: %s", member.PoolID)
	t.Logf("Address: %s", member.Address)
	t.Logf("ProtocolPort: %d", member.ProtocolPort)
	t.Logf("AdminStateUp: %t", member.AdminStateUp)
}

// PrintMonitor will print a monitor and all of its attributes.
func PrintMonitor(t *testing.T, monitor *monitors.Monitor) {
	t.Logf("ID: %s", monitor.ID)
	t.Logf("Name: %s", monitor.Name)
	t.Logf("TenantID: %s", monitor.TenantID)
	t.Logf("Type: %s", monitor.Type)
	t.Logf("Delay: %d", monitor.Delay)
	t.Logf("Timeout: %d", monitor.Timeout)
	t.Logf("MaxRetries: %d", monitor.MaxRetries)
	t.Logf("HTTPMethod: %s", monitor.HTTPMethod)
	t.Logf("URLPath: %s", monitor.URLPath)
	t.Logf("ExpectedCodes: %s", monitor.ExpectedCodes)
	t.Logf("AdminStateUp: %t", monitor.AdminStateUp)
	t.Logf("Status: %s", monitor.Status)

	t.Logf("Pools")
	for _, pool := range monitor.Pools {
		t.Logf("\t%#v", pool)
	}
}

// PrintPool will print a pool and all of its attributes.
func PrintPool(t *testing.T, pool *pools.Pool) {
	t.Logf("ID: %s", pool.ID)
	t.Logf("Name: %s", pool.Name)
	t.Logf("TenantID: %s", pool.TenantID)
	t.Logf("Description: %s", pool.Description)
	t.Logf("LBMethod: %s", pool.LBMethod)
	t.Logf("Protocol: %s", pool.Protocol)
	t.Logf("MonitorID: %s", pool.MonitorID)
	t.Logf("SubnetID: %s", pool.SubnetID)
	t.Logf("AdminStateUp: %t", pool.AdminStateUp)
	t.Logf("Persistence: %s", pool.Persistence)
	t.Logf("Provider: %s", pool.Provider)
	t.Logf("Monitor: %#v", pool.Monitor)

	t.Logf("Listeners")
	for _, listener := range pool.Listeners {
		t.Logf("\t%#v", listener)
	}

	t.Logf("Members")
	for _, member := range pool.Members {
		t.Logf("\t%#v", member)
	}

	t.Logf("Loadbalancers")
	for _, lb := range pool.Loadbalancers {
		t.Logf("\t%#v", lb)
	}
}

// WaitForLoadBalancerState will wait until a loadbalancer reaches a given state.
func WaitForLoadBalancerState(client *gophercloud.ServiceClient, lbID, status string, secs int) error {
	return gophercloud.WaitFor(secs, func() (bool, error) {
		current, err := loadbalancers.Get(client, lbID).Extract()
		if err != nil {
			if httpStatus, ok := err.(gophercloud.ErrDefault404); ok {
				if httpStatus.Actual == 404 {
					if status == "DELETED" {
						return true, nil
					}
				}
			}
			return false, err
		}

		if current.ProvisioningStatus == status {
			return true, nil
		}

		return false, nil
	})
}
