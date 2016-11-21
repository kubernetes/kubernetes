// +build acceptance compute defsecrules

package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	dsr "github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/defsecrules"
)

func TestDefSecRulesList(t *testing.T) {
	client, err := clients.NewComputeV2Client()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	allPages, err := dsr.List(client).AllPages()
	if err != nil {
		t.Fatalf("Unable to list default rules: %v", err)
	}

	allDefaultRules, err := dsr.ExtractDefaultRules(allPages)
	if err != nil {
		t.Fatalf("Unable to extract default rules: %v", err)
	}

	for _, defaultRule := range allDefaultRules {
		PrintDefaultRule(t, &defaultRule)
	}
}

func TestDefSecRulesCreate(t *testing.T) {
	client, err := clients.NewComputeV2Client()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	defaultRule, err := CreateDefaultRule(t, client)
	if err != nil {
		t.Fatalf("Unable to create default rule: %v", err)
	}
	defer DeleteDefaultRule(t, client, defaultRule)

	PrintDefaultRule(t, &defaultRule)
}

func TestDefSecRulesGet(t *testing.T) {
	client, err := clients.NewComputeV2Client()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	defaultRule, err := CreateDefaultRule(t, client)
	if err != nil {
		t.Fatalf("Unable to create default rule: %v", err)
	}
	defer DeleteDefaultRule(t, client, defaultRule)

	newDefaultRule, err := dsr.Get(client, defaultRule.ID).Extract()
	if err != nil {
		t.Fatalf("Unable to get default rule %s: %v", defaultRule.ID, err)
	}

	PrintDefaultRule(t, newDefaultRule)
}
