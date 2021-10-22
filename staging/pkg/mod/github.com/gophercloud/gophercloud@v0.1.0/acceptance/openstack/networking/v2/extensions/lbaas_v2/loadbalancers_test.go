// +build acceptance networking lbaas_v2 loadbalancers

package lbaas_v2

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	networking "github.com/gophercloud/gophercloud/acceptance/openstack/networking/v2"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/lbaas_v2/l7policies"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/lbaas_v2/listeners"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/lbaas_v2/loadbalancers"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/lbaas_v2/monitors"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/lbaas_v2/pools"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestLoadbalancersList(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	th.AssertNoErr(t, err)

	allPages, err := loadbalancers.List(client, nil).AllPages()
	th.AssertNoErr(t, err)

	allLoadbalancers, err := loadbalancers.ExtractLoadBalancers(allPages)
	th.AssertNoErr(t, err)

	for _, lb := range allLoadbalancers {
		tools.PrintResource(t, lb)
	}
}

func TestLoadbalancersCRUD(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	th.AssertNoErr(t, err)

	network, err := networking.CreateNetwork(t, client)
	th.AssertNoErr(t, err)
	defer networking.DeleteNetwork(t, client, network.ID)

	subnet, err := networking.CreateSubnet(t, client, network.ID)
	th.AssertNoErr(t, err)
	defer networking.DeleteSubnet(t, client, subnet.ID)

	lb, err := CreateLoadBalancer(t, client, subnet.ID)
	th.AssertNoErr(t, err)
	defer DeleteLoadBalancer(t, client, lb.ID)

	lbDescription := ""
	updateLoadBalancerOpts := loadbalancers.UpdateOpts{
		Description: &lbDescription,
	}
	_, err = loadbalancers.Update(client, lb.ID, updateLoadBalancerOpts).Extract()
	th.AssertNoErr(t, err)

	if err := WaitForLoadBalancerState(client, lb.ID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		t.Fatalf("Timed out waiting for loadbalancer to become active")
	}

	newLB, err := loadbalancers.Get(client, lb.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newLB)

	th.AssertEquals(t, newLB.Description, lbDescription)

	lbStats, err := loadbalancers.GetStats(client, lb.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, lbStats)

	// Because of the time it takes to create a loadbalancer,
	// this test will include some other resources.

	// Listener
	listener, err := CreateListener(t, client, lb)
	th.AssertNoErr(t, err)
	defer DeleteListener(t, client, lb.ID, listener.ID)

	listenerName := ""
	listenerDescription := ""
	updateListenerOpts := listeners.UpdateOpts{
		Name:        &listenerName,
		Description: &listenerDescription,
	}
	_, err = listeners.Update(client, listener.ID, updateListenerOpts).Extract()
	th.AssertNoErr(t, err)

	if err := WaitForLoadBalancerState(client, lb.ID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		t.Fatalf("Timed out waiting for loadbalancer to become active")
	}

	newListener, err := listeners.Get(client, listener.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newListener)

	th.AssertEquals(t, newListener.Name, listenerName)
	th.AssertEquals(t, newListener.Description, listenerDescription)

	// L7 policy
	policy, err := CreateL7Policy(t, client, listener, lb)
	th.AssertNoErr(t, err)
	defer DeleteL7Policy(t, client, lb.ID, policy.ID)

	newDescription := ""
	updateL7policyOpts := l7policies.UpdateOpts{
		Description: &newDescription,
	}
	_, err = l7policies.Update(client, policy.ID, updateL7policyOpts).Extract()
	th.AssertNoErr(t, err)

	if err := WaitForLoadBalancerState(client, lb.ID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		t.Fatalf("Timed out waiting for loadbalancer to become active")
	}

	newPolicy, err := l7policies.Get(client, policy.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newPolicy)

	th.AssertEquals(t, newPolicy.Description, newDescription)

	// L7 rule
	rule, err := CreateL7Rule(t, client, newPolicy.ID, lb)
	th.AssertNoErr(t, err)
	defer DeleteL7Rule(t, client, lb.ID, policy.ID, rule.ID)

	allPages, err := l7policies.ListRules(client, policy.ID, l7policies.ListRulesOpts{}).AllPages()
	th.AssertNoErr(t, err)
	allRules, err := l7policies.ExtractRules(allPages)
	th.AssertNoErr(t, err)
	for _, rule := range allRules {
		tools.PrintResource(t, rule)
	}

	/* NOT supported on F5 driver */
	updateL7ruleOpts := l7policies.UpdateRuleOpts{
		RuleType:    l7policies.TypePath,
		CompareType: l7policies.CompareTypeRegex,
		Value:       "/images/special*",
	}
	_, err = l7policies.UpdateRule(client, policy.ID, rule.ID, updateL7ruleOpts).Extract()
	th.AssertNoErr(t, err)

	if err := WaitForLoadBalancerState(client, lb.ID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		t.Fatalf("Timed out waiting for loadbalancer to become active")
	}

	newRule, err := l7policies.GetRule(client, newPolicy.ID, rule.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newRule)

	// Pool
	pool, err := CreatePool(t, client, lb)
	th.AssertNoErr(t, err)
	defer DeletePool(t, client, lb.ID, pool.ID)

	poolName := ""
	poolDescription := ""
	updatePoolOpts := pools.UpdateOpts{
		Name:        &poolName,
		Description: &poolDescription,
	}
	_, err = pools.Update(client, pool.ID, updatePoolOpts).Extract()
	th.AssertNoErr(t, err)

	if err := WaitForLoadBalancerState(client, lb.ID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		t.Fatalf("Timed out waiting for loadbalancer to become active")
	}

	newPool, err := pools.Get(client, pool.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newPool)
	th.AssertEquals(t, newPool.Name, poolName)
	th.AssertEquals(t, newPool.Description, poolDescription)

	// Update L7policy to redirect to pool
	newRedirectURL := ""
	updateL7policyOpts = l7policies.UpdateOpts{
		Action:         l7policies.ActionRedirectToPool,
		RedirectPoolID: &newPool.ID,
		RedirectURL:    &newRedirectURL,
	}
	_, err = l7policies.Update(client, policy.ID, updateL7policyOpts).Extract()
	th.AssertNoErr(t, err)

	if err := WaitForLoadBalancerState(client, lb.ID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		t.Fatalf("Timed out waiting for loadbalancer to become active")
	}

	newPolicy, err = l7policies.Get(client, policy.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newPolicy)

	th.AssertEquals(t, newPolicy.Description, newDescription)
	th.AssertEquals(t, newPolicy.Action, string(l7policies.ActionRedirectToPool))
	th.AssertEquals(t, newPolicy.RedirectPoolID, newPool.ID)
	th.AssertEquals(t, newPolicy.RedirectURL, newRedirectURL)

	// Workaround for proper delete order
	defer DeleteL7Policy(t, client, lb.ID, policy.ID)
	defer DeleteL7Rule(t, client, lb.ID, policy.ID, rule.ID)

	// Update listener's default pool ID
	updateListenerOpts = listeners.UpdateOpts{
		DefaultPoolID: &pool.ID,
	}
	_, err = listeners.Update(client, listener.ID, updateListenerOpts).Extract()
	th.AssertNoErr(t, err)

	if err := WaitForLoadBalancerState(client, lb.ID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		t.Fatalf("Timed out waiting for loadbalancer to become active")
	}

	newListener, err = listeners.Get(client, listener.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newListener)

	th.AssertEquals(t, newListener.DefaultPoolID, pool.ID)

	// Remove listener's default pool ID
	emptyPoolID := ""
	updateListenerOpts = listeners.UpdateOpts{
		DefaultPoolID: &emptyPoolID,
	}
	_, err = listeners.Update(client, listener.ID, updateListenerOpts).Extract()
	th.AssertNoErr(t, err)

	if err := WaitForLoadBalancerState(client, lb.ID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		t.Fatalf("Timed out waiting for loadbalancer to become active")
	}

	newListener, err = listeners.Get(client, listener.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newListener)

	th.AssertEquals(t, newListener.DefaultPoolID, "")

	// Member
	member, err := CreateMember(t, client, lb, newPool, subnet.ID, subnet.CIDR)
	th.AssertNoErr(t, err)
	defer DeleteMember(t, client, lb.ID, pool.ID, member.ID)

	memberName := ""
	newWeight := tools.RandomInt(11, 100)
	updateMemberOpts := pools.UpdateMemberOpts{
		Name:   &memberName,
		Weight: &newWeight,
	}
	_, err = pools.UpdateMember(client, pool.ID, member.ID, updateMemberOpts).Extract()
	th.AssertNoErr(t, err)

	if err := WaitForLoadBalancerState(client, lb.ID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		t.Fatalf("Timed out waiting for loadbalancer to become active")
	}

	newMember, err := pools.GetMember(client, pool.ID, member.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newMember)
	th.AssertEquals(t, newMember.Name, memberName)
	th.AssertEquals(t, newMember.Weight, newWeight)

	// Monitor
	monitor, err := CreateMonitor(t, client, lb, newPool)
	th.AssertNoErr(t, err)
	defer DeleteMonitor(t, client, lb.ID, monitor.ID)

	monName := ""
	newDelay := tools.RandomInt(20, 30)
	updateMonitorOpts := monitors.UpdateOpts{
		Name:  &monName,
		Delay: newDelay,
	}
	_, err = monitors.Update(client, monitor.ID, updateMonitorOpts).Extract()
	th.AssertNoErr(t, err)

	if err := WaitForLoadBalancerState(client, lb.ID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		t.Fatalf("Timed out waiting for loadbalancer to become active")
	}

	newMonitor, err := monitors.Get(client, monitor.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newMonitor)

	th.AssertEquals(t, newMonitor.Name, monName)
	th.AssertEquals(t, newMonitor.Delay, newDelay)
}
