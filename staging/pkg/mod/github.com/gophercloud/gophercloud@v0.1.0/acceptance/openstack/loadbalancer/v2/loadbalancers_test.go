// +build acceptance networking loadbalancer loadbalancers

package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	networking "github.com/gophercloud/gophercloud/acceptance/openstack/networking/v2"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/loadbalancer/v2/l7policies"
	"github.com/gophercloud/gophercloud/openstack/loadbalancer/v2/listeners"
	"github.com/gophercloud/gophercloud/openstack/loadbalancer/v2/loadbalancers"
	"github.com/gophercloud/gophercloud/openstack/loadbalancer/v2/monitors"
	"github.com/gophercloud/gophercloud/openstack/loadbalancer/v2/pools"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestLoadbalancersList(t *testing.T) {
	client, err := clients.NewLoadBalancerV2Client()
	th.AssertNoErr(t, err)

	allPages, err := loadbalancers.List(client, nil).AllPages()
	th.AssertNoErr(t, err)

	allLoadbalancers, err := loadbalancers.ExtractLoadBalancers(allPages)
	th.AssertNoErr(t, err)

	for _, lb := range allLoadbalancers {
		tools.PrintResource(t, lb)
	}
}

func TestLoadbalancersListByTags(t *testing.T) {
	netClient, err := clients.NewNetworkV2Client()
	th.AssertNoErr(t, err)

	lbClient, err := clients.NewLoadBalancerV2Client()
	th.AssertNoErr(t, err)

	network, err := networking.CreateNetwork(t, netClient)
	th.AssertNoErr(t, err)
	defer networking.DeleteNetwork(t, netClient, network.ID)

	subnet, err := networking.CreateSubnet(t, netClient, network.ID)
	th.AssertNoErr(t, err)
	defer networking.DeleteSubnet(t, netClient, subnet.ID)

	// Add "test" tag intentionally to test the "not-tags" parameter. Because "test" tag is also used in other test
	// cases, we use "test" tag to exclude load balancers created by other test case.
	tags := []string{"tag1", "tag2", "test"}
	lb, err := CreateLoadBalancer(t, lbClient, subnet.ID, tags)
	th.AssertNoErr(t, err)
	defer DeleteLoadBalancer(t, lbClient, lb.ID)

	tags = []string{"tag1"}
	listOpts := loadbalancers.ListOpts{
		Tags: tags,
	}
	allPages, err := loadbalancers.List(lbClient, listOpts).AllPages()
	th.AssertNoErr(t, err)
	allLoadbalancers, err := loadbalancers.ExtractLoadBalancers(allPages)
	th.AssertNoErr(t, err)
	th.AssertEquals(t, 1, len(allLoadbalancers))

	tags = []string{"test"}
	listOpts = loadbalancers.ListOpts{
		TagsNot: tags,
	}
	allPages, err = loadbalancers.List(lbClient, listOpts).AllPages()
	th.AssertNoErr(t, err)
	allLoadbalancers, err = loadbalancers.ExtractLoadBalancers(allPages)
	th.AssertNoErr(t, err)
	th.AssertEquals(t, 0, len(allLoadbalancers))

	tags = []string{"tag1", "tag3"}
	listOpts = loadbalancers.ListOpts{
		TagsAny: tags,
	}
	allPages, err = loadbalancers.List(lbClient, listOpts).AllPages()
	th.AssertNoErr(t, err)
	allLoadbalancers, err = loadbalancers.ExtractLoadBalancers(allPages)
	th.AssertNoErr(t, err)
	th.AssertEquals(t, 1, len(allLoadbalancers))

	tags = []string{"tag1", "test"}
	listOpts = loadbalancers.ListOpts{
		TagsNotAny: tags,
	}
	allPages, err = loadbalancers.List(lbClient, listOpts).AllPages()
	th.AssertNoErr(t, err)
	allLoadbalancers, err = loadbalancers.ExtractLoadBalancers(allPages)
	th.AssertNoErr(t, err)
	th.AssertEquals(t, 0, len(allLoadbalancers))
}

func TestLoadbalancersCRUD(t *testing.T) {
	netClient, err := clients.NewNetworkV2Client()
	th.AssertNoErr(t, err)

	lbClient, err := clients.NewLoadBalancerV2Client()
	th.AssertNoErr(t, err)

	network, err := networking.CreateNetwork(t, netClient)
	th.AssertNoErr(t, err)
	defer networking.DeleteNetwork(t, netClient, network.ID)

	subnet, err := networking.CreateSubnet(t, netClient, network.ID)
	th.AssertNoErr(t, err)
	defer networking.DeleteSubnet(t, netClient, subnet.ID)

	tags := []string{"test"}
	lb, err := CreateLoadBalancer(t, lbClient, subnet.ID, tags)
	th.AssertNoErr(t, err)
	defer DeleteLoadBalancer(t, lbClient, lb.ID)

	lbDescription := ""
	updateLoadBalancerOpts := loadbalancers.UpdateOpts{
		Description: &lbDescription,
	}
	_, err = loadbalancers.Update(lbClient, lb.ID, updateLoadBalancerOpts).Extract()
	th.AssertNoErr(t, err)

	if err = WaitForLoadBalancerState(lbClient, lb.ID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		t.Fatalf("Timed out waiting for loadbalancer to become active")
	}

	newLB, err := loadbalancers.Get(lbClient, lb.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newLB)

	th.AssertEquals(t, newLB.Description, lbDescription)

	lbStats, err := loadbalancers.GetStats(lbClient, lb.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, lbStats)

	// Because of the time it takes to create a loadbalancer,
	// this test will include some other resources.

	// Listener
	listener, err := CreateListener(t, lbClient, lb)
	th.AssertNoErr(t, err)
	defer DeleteListener(t, lbClient, lb.ID, listener.ID)

	listenerName := ""
	listenerDescription := ""
	updateListenerOpts := listeners.UpdateOpts{
		Name:        &listenerName,
		Description: &listenerDescription,
	}
	_, err = listeners.Update(lbClient, listener.ID, updateListenerOpts).Extract()
	th.AssertNoErr(t, err)

	if err = WaitForLoadBalancerState(lbClient, lb.ID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		t.Fatalf("Timed out waiting for loadbalancer to become active")
	}

	newListener, err := listeners.Get(lbClient, listener.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newListener)

	th.AssertEquals(t, newListener.Name, listenerName)
	th.AssertEquals(t, newListener.Description, listenerDescription)

	listenerStats, err := listeners.GetStats(lbClient, listener.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, listenerStats)

	// L7 policy
	policy, err := CreateL7Policy(t, lbClient, listener, lb)
	th.AssertNoErr(t, err)
	defer DeleteL7Policy(t, lbClient, lb.ID, policy.ID)

	newDescription := ""
	updateL7policyOpts := l7policies.UpdateOpts{
		Description: &newDescription,
	}
	_, err = l7policies.Update(lbClient, policy.ID, updateL7policyOpts).Extract()
	th.AssertNoErr(t, err)

	if err = WaitForLoadBalancerState(lbClient, lb.ID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		t.Fatalf("Timed out waiting for loadbalancer to become active")
	}

	newPolicy, err := l7policies.Get(lbClient, policy.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newPolicy)

	th.AssertEquals(t, newPolicy.Description, newDescription)

	// L7 rule
	rule, err := CreateL7Rule(t, lbClient, newPolicy.ID, lb)
	th.AssertNoErr(t, err)
	defer DeleteL7Rule(t, lbClient, lb.ID, policy.ID, rule.ID)

	allPages, err := l7policies.ListRules(lbClient, policy.ID, l7policies.ListRulesOpts{}).AllPages()
	th.AssertNoErr(t, err)
	allRules, err := l7policies.ExtractRules(allPages)
	th.AssertNoErr(t, err)
	for _, rule := range allRules {
		tools.PrintResource(t, rule)
	}

	updateL7ruleOpts := l7policies.UpdateRuleOpts{
		RuleType:    l7policies.TypePath,
		CompareType: l7policies.CompareTypeRegex,
		Value:       "/images/special*",
	}
	_, err = l7policies.UpdateRule(lbClient, policy.ID, rule.ID, updateL7ruleOpts).Extract()
	th.AssertNoErr(t, err)

	if err = WaitForLoadBalancerState(lbClient, lb.ID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		t.Fatalf("Timed out waiting for loadbalancer to become active")
	}

	newRule, err := l7policies.GetRule(lbClient, newPolicy.ID, rule.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newRule)

	// Pool
	pool, err := CreatePool(t, lbClient, lb)
	th.AssertNoErr(t, err)
	defer DeletePool(t, lbClient, lb.ID, pool.ID)

	poolName := ""
	poolDescription := ""
	updatePoolOpts := pools.UpdateOpts{
		Name:        &poolName,
		Description: &poolDescription,
	}
	_, err = pools.Update(lbClient, pool.ID, updatePoolOpts).Extract()
	th.AssertNoErr(t, err)

	if err = WaitForLoadBalancerState(lbClient, lb.ID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		t.Fatalf("Timed out waiting for loadbalancer to become active")
	}

	newPool, err := pools.Get(lbClient, pool.ID).Extract()
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
	_, err = l7policies.Update(lbClient, policy.ID, updateL7policyOpts).Extract()
	th.AssertNoErr(t, err)

	if err := WaitForLoadBalancerState(lbClient, lb.ID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		t.Fatalf("Timed out waiting for loadbalancer to become active")
	}

	newPolicy, err = l7policies.Get(lbClient, policy.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newPolicy)

	th.AssertEquals(t, newPolicy.Description, newDescription)
	th.AssertEquals(t, newPolicy.Action, string(l7policies.ActionRedirectToPool))
	th.AssertEquals(t, newPolicy.RedirectPoolID, newPool.ID)
	th.AssertEquals(t, newPolicy.RedirectURL, newRedirectURL)

	// Workaround for proper delete order
	defer DeleteL7Policy(t, lbClient, lb.ID, policy.ID)
	defer DeleteL7Rule(t, lbClient, lb.ID, policy.ID, rule.ID)

	// Update listener's default pool ID
	updateListenerOpts = listeners.UpdateOpts{
		DefaultPoolID: &pool.ID,
	}
	_, err = listeners.Update(lbClient, listener.ID, updateListenerOpts).Extract()
	th.AssertNoErr(t, err)

	if err := WaitForLoadBalancerState(lbClient, lb.ID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		t.Fatalf("Timed out waiting for loadbalancer to become active")
	}

	newListener, err = listeners.Get(lbClient, listener.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newListener)

	th.AssertEquals(t, newListener.DefaultPoolID, pool.ID)

	// Remove listener's default pool ID
	emptyPoolID := ""
	updateListenerOpts = listeners.UpdateOpts{
		DefaultPoolID: &emptyPoolID,
	}
	_, err = listeners.Update(lbClient, listener.ID, updateListenerOpts).Extract()
	th.AssertNoErr(t, err)

	if err := WaitForLoadBalancerState(lbClient, lb.ID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		t.Fatalf("Timed out waiting for loadbalancer to become active")
	}

	newListener, err = listeners.Get(lbClient, listener.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newListener)

	th.AssertEquals(t, newListener.DefaultPoolID, "")

	// Member
	member, err := CreateMember(t, lbClient, lb, newPool, subnet.ID, subnet.CIDR)
	th.AssertNoErr(t, err)
	defer DeleteMember(t, lbClient, lb.ID, pool.ID, member.ID)

	memberName := ""
	newWeight := tools.RandomInt(11, 100)
	updateMemberOpts := pools.UpdateMemberOpts{
		Name:   &memberName,
		Weight: &newWeight,
	}
	_, err = pools.UpdateMember(lbClient, pool.ID, member.ID, updateMemberOpts).Extract()
	th.AssertNoErr(t, err)

	if err = WaitForLoadBalancerState(lbClient, lb.ID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		t.Fatalf("Timed out waiting for loadbalancer to become active")
	}

	newMember, err := pools.GetMember(lbClient, pool.ID, member.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newMember)
	th.AssertEquals(t, newMember.Name, memberName)

	newWeight = tools.RandomInt(11, 100)
	memberOpts := pools.BatchUpdateMemberOpts{
		Address:      member.Address,
		ProtocolPort: member.ProtocolPort,
		Weight:       &newWeight,
	}
	batchMembers := []pools.BatchUpdateMemberOpts{memberOpts}
	if err := pools.BatchUpdateMembers(lbClient, pool.ID, batchMembers).ExtractErr(); err != nil {
		t.Fatalf("Unable to batch update members")
	}

	if err = WaitForLoadBalancerState(lbClient, lb.ID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		t.Fatalf("Timed out waiting for loadbalancer to become active")
	}

	newMember, err = pools.GetMember(lbClient, pool.ID, member.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newMember)

	// Monitor
	monitor, err := CreateMonitor(t, lbClient, lb, newPool)
	th.AssertNoErr(t, err)
	defer DeleteMonitor(t, lbClient, lb.ID, monitor.ID)

	monName := ""
	newDelay := tools.RandomInt(20, 30)
	updateMonitorOpts := monitors.UpdateOpts{
		Name:  &monName,
		Delay: newDelay,
	}
	_, err = monitors.Update(lbClient, monitor.ID, updateMonitorOpts).Extract()
	th.AssertNoErr(t, err)

	if err = WaitForLoadBalancerState(lbClient, lb.ID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		t.Fatalf("Timed out waiting for loadbalancer to become active")
	}

	newMonitor, err := monitors.Get(lbClient, monitor.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newMonitor)

	th.AssertEquals(t, newMonitor.Name, monName)
	th.AssertEquals(t, newMonitor.Delay, newDelay)
}

func TestLoadbalancersCascadeCRUD(t *testing.T) {
	netClient, err := clients.NewNetworkV2Client()
	th.AssertNoErr(t, err)

	lbClient, err := clients.NewLoadBalancerV2Client()
	th.AssertNoErr(t, err)

	network, err := networking.CreateNetwork(t, netClient)
	th.AssertNoErr(t, err)
	defer networking.DeleteNetwork(t, netClient, network.ID)

	subnet, err := networking.CreateSubnet(t, netClient, network.ID)
	th.AssertNoErr(t, err)
	defer networking.DeleteSubnet(t, netClient, subnet.ID)

	tags := []string{"test"}
	lb, err := CreateLoadBalancer(t, lbClient, subnet.ID, tags)
	th.AssertNoErr(t, err)
	defer CascadeDeleteLoadBalancer(t, lbClient, lb.ID)

	newLB, err := loadbalancers.Get(lbClient, lb.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newLB)

	// Because of the time it takes to create a loadbalancer,
	// this test will include some other resources.

	// Listener
	listener, err := CreateListener(t, lbClient, lb)
	th.AssertNoErr(t, err)

	listenerDescription := "Some listener description"
	updateListenerOpts := listeners.UpdateOpts{
		Description: &listenerDescription,
	}
	_, err = listeners.Update(lbClient, listener.ID, updateListenerOpts).Extract()
	th.AssertNoErr(t, err)

	if err := WaitForLoadBalancerState(lbClient, lb.ID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		t.Fatalf("Timed out waiting for loadbalancer to become active")
	}

	newListener, err := listeners.Get(lbClient, listener.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newListener)

	// Pool
	pool, err := CreatePool(t, lbClient, lb)
	th.AssertNoErr(t, err)

	poolDescription := "Some pool description"
	updatePoolOpts := pools.UpdateOpts{
		Description: &poolDescription,
	}
	_, err = pools.Update(lbClient, pool.ID, updatePoolOpts).Extract()
	th.AssertNoErr(t, err)

	if err := WaitForLoadBalancerState(lbClient, lb.ID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		t.Fatalf("Timed out waiting for loadbalancer to become active")
	}

	newPool, err := pools.Get(lbClient, pool.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newPool)

	// Member
	member, err := CreateMember(t, lbClient, lb, newPool, subnet.ID, subnet.CIDR)
	th.AssertNoErr(t, err)

	newWeight := tools.RandomInt(11, 100)
	updateMemberOpts := pools.UpdateMemberOpts{
		Weight: &newWeight,
	}
	_, err = pools.UpdateMember(lbClient, pool.ID, member.ID, updateMemberOpts).Extract()
	th.AssertNoErr(t, err)

	if err := WaitForLoadBalancerState(lbClient, lb.ID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		t.Fatalf("Timed out waiting for loadbalancer to become active")
	}

	newMember, err := pools.GetMember(lbClient, pool.ID, member.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newMember)

	// Monitor
	monitor, err := CreateMonitor(t, lbClient, lb, newPool)
	th.AssertNoErr(t, err)

	newDelay := tools.RandomInt(20, 30)
	updateMonitorOpts := monitors.UpdateOpts{
		Delay: newDelay,
	}
	_, err = monitors.Update(lbClient, monitor.ID, updateMonitorOpts).Extract()
	th.AssertNoErr(t, err)

	if err := WaitForLoadBalancerState(lbClient, lb.ID, "ACTIVE", loadbalancerActiveTimeoutSeconds); err != nil {
		t.Fatalf("Timed out waiting for loadbalancer to become active")
	}

	newMonitor, err := monitors.Get(lbClient, monitor.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newMonitor)

}
