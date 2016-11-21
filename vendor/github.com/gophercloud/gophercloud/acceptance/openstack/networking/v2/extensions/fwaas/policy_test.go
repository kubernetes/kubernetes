// +build acceptance networking fwaas

package fwaas

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/fwaas/policies"
)

func TestPolicyList(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	if err != nil {
		t.Fatalf("Unable to create a network client: %v", err)
	}

	allPages, err := policies.List(client, nil).AllPages()
	if err != nil {
		t.Fatalf("Unable to list policies: %v", err)
	}

	allPolicies, err := policies.ExtractPolicies(allPages)
	if err != nil {
		t.Fatalf("Unable to extract policies: %v", err)
	}

	for _, policy := range allPolicies {
		PrintPolicy(t, &policy)
	}
}

func TestPolicyCRUD(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	if err != nil {
		t.Fatalf("Unable to create a network client: %v", err)
	}

	rule, err := CreateRule(t, client)
	if err != nil {
		t.Fatalf("Unable to create rule: %v", err)
	}
	defer DeleteRule(t, client, rule.ID)

	PrintRule(t, rule)

	policy, err := CreatePolicy(t, client, rule.ID)
	if err != nil {
		t.Fatalf("Unable to create policy: %v", err)
	}
	defer DeletePolicy(t, client, policy.ID)

	PrintPolicy(t, policy)

	updateOpts := policies.UpdateOpts{
		Description: "Some policy description",
	}

	_, err = policies.Update(client, policy.ID, updateOpts).Extract()
	if err != nil {
		t.Fatalf("Unable to update policy: %v", err)
	}

	newPolicy, err := policies.Get(client, policy.ID).Extract()
	if err != nil {
		t.Fatalf("Unable to get policy: %v", err)
	}

	PrintPolicy(t, newPolicy)
}
