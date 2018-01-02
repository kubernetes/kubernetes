// +build acceptance networking fwaas

package fwaas

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/fwaas/rules"
)

func TestRuleList(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	if err != nil {
		t.Fatalf("Unable to create a network client: %v", err)
	}

	allPages, err := rules.List(client, nil).AllPages()
	if err != nil {
		t.Fatalf("Unable to list rules: %v", err)
	}

	allRules, err := rules.ExtractRules(allPages)
	if err != nil {
		t.Fatalf("Unable to extract rules: %v", err)
	}

	for _, rule := range allRules {
		tools.PrintResource(t, rule)
	}
}

func TestRuleCRUD(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	if err != nil {
		t.Fatalf("Unable to create a network client: %v", err)
	}

	rule, err := CreateRule(t, client)
	if err != nil {
		t.Fatalf("Unable to create rule: %v", err)
	}
	defer DeleteRule(t, client, rule.ID)

	tools.PrintResource(t, rule)

	ruleDescription := "Some rule description"
	updateOpts := rules.UpdateOpts{
		Description: &ruleDescription,
	}

	_, err = rules.Update(client, rule.ID, updateOpts).Extract()
	if err != nil {
		t.Fatalf("Unable to update rule: %v", err)
	}

	newRule, err := rules.Get(client, rule.ID).Extract()
	if err != nil {
		t.Fatalf("Unable to get rule: %v", err)
	}

	tools.PrintResource(t, newRule)
}
