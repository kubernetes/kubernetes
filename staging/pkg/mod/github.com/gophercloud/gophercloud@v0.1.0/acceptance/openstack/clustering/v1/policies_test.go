// +build acceptance clustering policies

package v1

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/clustering/v1/policies"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestPoliciesCRUD(t *testing.T) {
	client, err := clients.NewClusteringV1Client()
	th.AssertNoErr(t, err)
	client.Microversion = "1.5"

	policy, err := CreatePolicy(t, client)
	th.AssertNoErr(t, err)
	defer DeletePolicy(t, client, policy.ID)

	// Test listing policies
	allPages, err := policies.List(client, nil).AllPages()
	th.AssertNoErr(t, err)

	allPolicies, err := policies.ExtractPolicies(allPages)
	th.AssertNoErr(t, err)

	var found bool
	for _, v := range allPolicies {
		if v.ID == policy.ID {
			found = true
		}
	}

	th.AssertEquals(t, found, true)

	// Test Get policy
	getPolicy, err := policies.Get(client, policy.ID).Extract()
	th.AssertNoErr(t, err)
	tools.PrintResource(t, getPolicy)

	// Test updating policy
	updateOpts := policies.UpdateOpts{
		Name: policy.Name + "-UPDATE",
	}

	t.Logf("Attempting to update policy: %s", policy.ID)
	updatePolicy, err := policies.Update(client, policy.ID, updateOpts).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, updatePolicy)
	tools.PrintResource(t, updatePolicy.UpdatedAt)

	// Test validating policy
	t.Logf("Attempting to validate policy: %s", policy.ID)
	validateOpts := policies.ValidateOpts{
		Spec: TestPolicySpec,
	}

	validatePolicy, err := policies.Validate(client, validateOpts).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, validatePolicy)

	th.AssertEquals(t, validatePolicy.Name, "validated_policy")
	th.AssertEquals(t, validatePolicy.Spec.Version, TestPolicySpec.Version)
}
