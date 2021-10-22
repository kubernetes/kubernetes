package v2

import (
	"fmt"
	"strings"
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/loadbalancer/v2/l7policies"
	"github.com/gophercloud/gophercloud/openstack/loadbalancer/v2/listeners"
	"github.com/gophercloud/gophercloud/openstack/loadbalancer/v2/loadbalancers"
	"github.com/gophercloud/gophercloud/openstack/loadbalancer/v2/monitors"
	"github.com/gophercloud/gophercloud/openstack/loadbalancer/v2/pools"
	th "github.com/gophercloud/gophercloud/testhelper"
)

const loadbalancerActiveTimeoutSeconds = 600
const loadbalancerDeleteTimeoutSeconds = 600

// CreateListener will create a listener for a given load balancer on a random
// port with a random name. An error will be returned if the listener could not
// be created.
func CreateListener(t *testing.T, client *gophercloud.ServiceClient, lb *loadbalancers.LoadBalancer) (*listeners.Listener, error) {
	listenerName := tools.RandomString("TESTACCT-", 8)
	listenerDescription := tools.RandomString("TESTACCT-DESC-", 8)
	listenerPort := tools.RandomInt(1, 100)

	t.Logf("Attempting to create listener %s on port %d", listenerName, listenerPort)

	createOpts := listeners.CreateOpts{
		Name:           listenerName,
		Description:    listenerDescription,
		LoadbalancerID: lb.ID,
		Protocol:       listeners.ProtocolTCP,
		ProtocolPort:   listenerPort,
	}

	listener, err := listeners.Create(client, createOpts).Extract()
	if err != nil {
		return listener, err
	}

	t.Logf("Successfully created listener %s", listenerName)

	if err := WaitForLoadBalancerState(client, lb.ID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		return listener, fmt.Errorf("Timed out waiting for loadbalancer to become active: %s", err)
	}

	th.AssertEquals(t, listener.Name, listenerName)
	th.AssertEquals(t, listener.Description, listenerDescription)
	th.AssertEquals(t, listener.Loadbalancers[0].ID, lb.ID)
	th.AssertEquals(t, listener.Protocol, string(listeners.ProtocolTCP))
	th.AssertEquals(t, listener.ProtocolPort, listenerPort)

	return listener, nil
}

// CreateLoadBalancer will create a load balancer with a random name on a given
// subnet. An error will be returned if the loadbalancer could not be created.
func CreateLoadBalancer(t *testing.T, client *gophercloud.ServiceClient, subnetID string, tags []string) (*loadbalancers.LoadBalancer, error) {
	lbName := tools.RandomString("TESTACCT-", 8)
	lbDescription := tools.RandomString("TESTACCT-DESC-", 8)

	t.Logf("Attempting to create loadbalancer %s on subnet %s", lbName, subnetID)

	createOpts := loadbalancers.CreateOpts{
		Name:         lbName,
		Description:  lbDescription,
		VipSubnetID:  subnetID,
		AdminStateUp: gophercloud.Enabled,
	}
	if len(tags) > 0 {
		createOpts.Tags = tags
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

	th.AssertEquals(t, lb.Name, lbName)
	th.AssertEquals(t, lb.Description, lbDescription)
	th.AssertEquals(t, lb.VipSubnetID, subnetID)
	th.AssertEquals(t, lb.AdminStateUp, true)

	if len(tags) > 0 {
		th.AssertDeepEquals(t, lb.Tags, tags)
	}

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
		Weight:       &memberWeight,
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
		return member, fmt.Errorf("Timed out waiting for loadbalancer to become active: %s", err)
	}

	th.AssertEquals(t, member.Name, memberName)

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
		Type:       monitors.TypePING,
	}

	monitor, err := monitors.Create(client, createOpts).Extract()
	if err != nil {
		return monitor, err
	}

	t.Logf("Successfully created monitor: %s", monitorName)

	if err := WaitForLoadBalancerState(client, lb.ID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		return monitor, fmt.Errorf("Timed out waiting for loadbalancer to become active: %s", err)
	}

	th.AssertEquals(t, monitor.Name, monitorName)
	th.AssertEquals(t, monitor.Type, monitors.TypePING)

	return monitor, nil
}

// CreatePool will create a pool with a random name with a specified listener
// and loadbalancer. An error will be returned if the pool could not be
// created.
func CreatePool(t *testing.T, client *gophercloud.ServiceClient, lb *loadbalancers.LoadBalancer) (*pools.Pool, error) {
	poolName := tools.RandomString("TESTACCT-", 8)
	poolDescription := tools.RandomString("TESTACCT-DESC-", 8)

	t.Logf("Attempting to create pool %s", poolName)

	createOpts := pools.CreateOpts{
		Name:           poolName,
		Description:    poolDescription,
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
		return pool, fmt.Errorf("Timed out waiting for loadbalancer to become active: %s", err)
	}

	th.AssertEquals(t, pool.Name, poolName)
	th.AssertEquals(t, pool.Description, poolDescription)
	th.AssertEquals(t, pool.Protocol, string(pools.ProtocolTCP))
	th.AssertEquals(t, pool.Loadbalancers[0].ID, lb.ID)
	th.AssertEquals(t, pool.LBMethod, string(pools.LBMethodLeastConnections))

	return pool, nil
}

// CreateL7Policy will create a l7 policy with a random name with a specified listener
// and loadbalancer. An error will be returned if the l7 policy could not be
// created.
func CreateL7Policy(t *testing.T, client *gophercloud.ServiceClient, listener *listeners.Listener, lb *loadbalancers.LoadBalancer) (*l7policies.L7Policy, error) {
	policyName := tools.RandomString("TESTACCT-", 8)
	policyDescription := tools.RandomString("TESTACCT-DESC-", 8)

	t.Logf("Attempting to create l7 policy %s", policyName)

	createOpts := l7policies.CreateOpts{
		Name:        policyName,
		Description: policyDescription,
		ListenerID:  listener.ID,
		Action:      l7policies.ActionRedirectToURL,
		RedirectURL: "http://www.example.com",
	}

	policy, err := l7policies.Create(client, createOpts).Extract()
	if err != nil {
		return policy, err
	}

	t.Logf("Successfully created l7 policy %s", policyName)

	if err := WaitForLoadBalancerState(client, lb.ID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		return policy, fmt.Errorf("Timed out waiting for loadbalancer to become active: %s", err)
	}

	th.AssertEquals(t, policy.Name, policyName)
	th.AssertEquals(t, policy.Description, policyDescription)
	th.AssertEquals(t, policy.ListenerID, listener.ID)
	th.AssertEquals(t, policy.Action, string(l7policies.ActionRedirectToURL))
	th.AssertEquals(t, policy.RedirectURL, "http://www.example.com")

	return policy, nil
}

// CreateL7Rule creates a l7 rule for specified l7 policy.
func CreateL7Rule(t *testing.T, client *gophercloud.ServiceClient, policyID string, lb *loadbalancers.LoadBalancer) (*l7policies.Rule, error) {
	t.Logf("Attempting to create l7 rule for policy %s", policyID)

	createOpts := l7policies.CreateRuleOpts{
		RuleType:    l7policies.TypePath,
		CompareType: l7policies.CompareTypeStartWith,
		Value:       "/api",
	}

	rule, err := l7policies.CreateRule(client, policyID, createOpts).Extract()
	if err != nil {
		return rule, err
	}

	t.Logf("Successfully created l7 rule for policy %s", policyID)

	if err := WaitForLoadBalancerState(client, lb.ID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		return rule, fmt.Errorf("Timed out waiting for loadbalancer to become active: %s", err)
	}

	th.AssertEquals(t, rule.RuleType, string(l7policies.TypePath))
	th.AssertEquals(t, rule.CompareType, string(l7policies.CompareTypeStartWith))
	th.AssertEquals(t, rule.Value, "/api")

	return rule, nil
}

// DeleteL7Policy will delete a specified l7 policy. A fatal error will occur if
// the l7 policy could not be deleted. This works best when used as a deferred
// function.
func DeleteL7Policy(t *testing.T, client *gophercloud.ServiceClient, lbID, policyID string) {
	t.Logf("Attempting to delete l7 policy %s", policyID)

	if err := l7policies.Delete(client, policyID).ExtractErr(); err != nil {
		if _, ok := err.(gophercloud.ErrDefault404); !ok {
			t.Fatalf("Unable to delete l7 policy: %v", err)
		}
	}

	if err := WaitForLoadBalancerState(client, lbID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		t.Fatalf("Timed out waiting for loadbalancer to become active: %s", err)
	}

	t.Logf("Successfully deleted l7 policy %s", policyID)
}

// DeleteL7Rule will delete a specified l7 rule. A fatal error will occur if
// the l7 rule could not be deleted. This works best when used as a deferred
// function.
func DeleteL7Rule(t *testing.T, client *gophercloud.ServiceClient, lbID, policyID, ruleID string) {
	t.Logf("Attempting to delete l7 rule %s", ruleID)

	if err := l7policies.DeleteRule(client, policyID, ruleID).ExtractErr(); err != nil {
		if _, ok := err.(gophercloud.ErrDefault404); !ok {
			t.Fatalf("Unable to delete l7 rule: %v", err)
		}
	}

	if err := WaitForLoadBalancerState(client, lbID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		t.Fatalf("Timed out waiting for loadbalancer to become active: %s", err)
	}

	t.Logf("Successfully deleted l7 rule %s", ruleID)
}

// DeleteListener will delete a specified listener. A fatal error will occur if
// the listener could not be deleted. This works best when used as a deferred
// function.
func DeleteListener(t *testing.T, client *gophercloud.ServiceClient, lbID, listenerID string) {
	t.Logf("Attempting to delete listener %s", listenerID)

	if err := listeners.Delete(client, listenerID).ExtractErr(); err != nil {
		if _, ok := err.(gophercloud.ErrDefault404); !ok {
			t.Fatalf("Unable to delete listener: %v", err)
		}
	}

	if err := WaitForLoadBalancerState(client, lbID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		t.Fatalf("Timed out waiting for loadbalancer to become active: %s", err)
	}

	t.Logf("Successfully deleted listener %s", listenerID)
}

// DeleteMember will delete a specified member. A fatal error will occur if the
// member could not be deleted. This works best when used as a deferred
// function.
func DeleteMember(t *testing.T, client *gophercloud.ServiceClient, lbID, poolID, memberID string) {
	t.Logf("Attempting to delete member %s", memberID)

	if err := pools.DeleteMember(client, poolID, memberID).ExtractErr(); err != nil {
		if _, ok := err.(gophercloud.ErrDefault404); !ok {
			t.Fatalf("Unable to delete member: %s", memberID)
		}
	}

	if err := WaitForLoadBalancerState(client, lbID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		t.Fatalf("Timed out waiting for loadbalancer to become active: %s", err)
	}

	t.Logf("Successfully deleted member %s", memberID)
}

// DeleteLoadBalancer will delete a specified loadbalancer. A fatal error will
// occur if the loadbalancer could not be deleted. This works best when used
// as a deferred function.
func DeleteLoadBalancer(t *testing.T, client *gophercloud.ServiceClient, lbID string) {
	t.Logf("Attempting to delete loadbalancer %s", lbID)

	deleteOpts := loadbalancers.DeleteOpts{
		Cascade: false,
	}

	if err := loadbalancers.Delete(client, lbID, deleteOpts).ExtractErr(); err != nil {
		if _, ok := err.(gophercloud.ErrDefault404); !ok {
			t.Fatalf("Unable to delete loadbalancer: %v", err)
		}
	}

	t.Logf("Waiting for loadbalancer %s to delete", lbID)

	if err := WaitForLoadBalancerState(client, lbID, "DELETED", loadbalancerActiveTimeoutSeconds); err != nil {
		t.Fatalf("Loadbalancer did not delete in time: %s", err)
	}

	t.Logf("Successfully deleted loadbalancer %s", lbID)
}

// CascadeDeleteLoadBalancer will perform a cascading delete on a loadbalancer.
// A fatal error will occur if the loadbalancer could not be deleted. This works
// best when used as a deferred function.
func CascadeDeleteLoadBalancer(t *testing.T, client *gophercloud.ServiceClient, lbID string) {
	t.Logf("Attempting to cascade delete loadbalancer %s", lbID)

	deleteOpts := loadbalancers.DeleteOpts{
		Cascade: true,
	}

	if err := loadbalancers.Delete(client, lbID, deleteOpts).ExtractErr(); err != nil {
		t.Fatalf("Unable to cascade delete loadbalancer: %v", err)
	}

	t.Logf("Waiting for loadbalancer %s to cascade delete", lbID)

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
		if _, ok := err.(gophercloud.ErrDefault404); !ok {
			t.Fatalf("Unable to delete monitor: %v", err)
		}
	}

	if err := WaitForLoadBalancerState(client, lbID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		t.Fatalf("Timed out waiting for loadbalancer to become active: %s", err)
	}

	t.Logf("Successfully deleted monitor %s", monitorID)
}

// DeletePool will delete a specified pool. A fatal error will occur if the
// pool could not be deleted. This works best when used as a deferred function.
func DeletePool(t *testing.T, client *gophercloud.ServiceClient, lbID, poolID string) {
	t.Logf("Attempting to delete pool %s", poolID)

	if err := pools.Delete(client, poolID).ExtractErr(); err != nil {
		if _, ok := err.(gophercloud.ErrDefault404); !ok {
			t.Fatalf("Unable to delete pool: %v", err)
		}
	}

	if err := WaitForLoadBalancerState(client, lbID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		t.Fatalf("Timed out waiting for loadbalancer to become active: %s", err)
	}

	t.Logf("Successfully deleted pool %s", poolID)
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

		if current.ProvisioningStatus == "ERROR" {
			return false, fmt.Errorf("Load balancer is in ERROR state")
		}

		return false, nil
	})
}
