// +build acceptance networking fwaas

package fwaas

import (
	"testing"

	base "github.com/rackspace/gophercloud/acceptance/openstack/networking/v2"
	"github.com/rackspace/gophercloud/openstack/networking/v2/extensions/fwaas/policies"
	"github.com/rackspace/gophercloud/openstack/networking/v2/extensions/fwaas/rules"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
)

func firewallPolicySetup(t *testing.T) string {
	base.Setup(t)
	return createRule(t, &rules.CreateOpts{
		Protocol: "tcp",
		Action:   "allow",
	})
}

func firewallPolicyTeardown(t *testing.T, ruleID string) {
	defer base.Teardown()
	deleteRule(t, ruleID)
}

func TestFirewallPolicy(t *testing.T) {
	ruleID := firewallPolicySetup(t)
	defer firewallPolicyTeardown(t, ruleID)

	policyID := createPolicy(t, &policies.CreateOpts{
		Name:        "gophercloud test",
		Description: "acceptance test",
		Rules: []string{
			ruleID,
		},
	})

	listPolicies(t)

	updatePolicy(t, policyID, &policies.UpdateOpts{
		Description: "acceptance test updated",
	})

	getPolicy(t, policyID)

	removeRuleFromPolicy(t, policyID, ruleID)

	addRuleToPolicy(t, policyID, ruleID)

	deletePolicy(t, policyID)
}

func createPolicy(t *testing.T, opts *policies.CreateOpts) string {
	p, err := policies.Create(base.Client, *opts).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Created policy: %#v", opts)
	return p.ID
}

func listPolicies(t *testing.T) {
	err := policies.List(base.Client, policies.ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		policyList, err := policies.ExtractPolicies(page)
		if err != nil {
			t.Errorf("Failed to extract policies: %v", err)
			return false, err
		}

		for _, p := range policyList {
			t.Logf("Listing policies: ID [%s]", p.ID)
		}

		return true, nil
	})
	th.AssertNoErr(t, err)
}

func updatePolicy(t *testing.T, policyID string, opts *policies.UpdateOpts) {
	p, err := policies.Update(base.Client, policyID, *opts).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Updated policy ID [%s]", p.ID)
}

func removeRuleFromPolicy(t *testing.T, policyID string, ruleID string) {
	err := policies.RemoveRule(base.Client, policyID, ruleID)
	th.AssertNoErr(t, err)
	t.Logf("Removed rule [%s] from policy ID [%s]", ruleID, policyID)
}

func addRuleToPolicy(t *testing.T, policyID string, ruleID string) {
	err := policies.InsertRule(base.Client, policyID, ruleID, "", "")
	th.AssertNoErr(t, err)
	t.Logf("Inserted rule [%s] into policy ID [%s]", ruleID, policyID)
}

func getPolicy(t *testing.T, policyID string) {
	p, err := policies.Get(base.Client, policyID).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Getting policy ID [%s]", p.ID)
}

func deletePolicy(t *testing.T, policyID string) {
	res := policies.Delete(base.Client, policyID)
	th.AssertNoErr(t, res.Err)
	t.Logf("Deleted policy %s", policyID)
}
