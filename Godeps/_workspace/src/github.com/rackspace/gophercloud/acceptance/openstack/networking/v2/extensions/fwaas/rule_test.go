// +build acceptance networking fwaas

package fwaas

import (
	"testing"

	base "github.com/rackspace/gophercloud/acceptance/openstack/networking/v2"
	"github.com/rackspace/gophercloud/openstack/networking/v2/extensions/fwaas/rules"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestFirewallRules(t *testing.T) {
	base.Setup(t)
	defer base.Teardown()

	ruleID := createRule(t, &rules.CreateOpts{
		Name:                 "gophercloud_test",
		Description:          "acceptance test",
		Protocol:             "tcp",
		Action:               "allow",
		DestinationIPAddress: "192.168.0.0/24",
		DestinationPort:      "22",
	})

	listRules(t)

	destinationIPAddress := "192.168.1.0/24"
	destinationPort := ""
	sourcePort := "1234"

	updateRule(t, ruleID, &rules.UpdateOpts{
		DestinationIPAddress: &destinationIPAddress,
		DestinationPort:      &destinationPort,
		SourcePort:           &sourcePort,
	})

	getRule(t, ruleID)

	deleteRule(t, ruleID)
}

func createRule(t *testing.T, opts *rules.CreateOpts) string {
	r, err := rules.Create(base.Client, *opts).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Created rule: %#v", opts)
	return r.ID
}

func listRules(t *testing.T) {
	err := rules.List(base.Client, rules.ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		ruleList, err := rules.ExtractRules(page)
		if err != nil {
			t.Errorf("Failed to extract rules: %v", err)
			return false, err
		}

		for _, r := range ruleList {
			t.Logf("Listing rules: ID [%s]", r.ID)
		}

		return true, nil
	})
	th.AssertNoErr(t, err)
}

func updateRule(t *testing.T, ruleID string, opts *rules.UpdateOpts) {
	r, err := rules.Update(base.Client, ruleID, *opts).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Updated rule ID [%s]", r.ID)
}

func getRule(t *testing.T, ruleID string) {
	r, err := rules.Get(base.Client, ruleID).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Getting rule ID [%s]", r.ID)
}

func deleteRule(t *testing.T, ruleID string) {
	res := rules.Delete(base.Client, ruleID)
	th.AssertNoErr(t, res.Err)
	t.Logf("Deleted rule %s", ruleID)
}
