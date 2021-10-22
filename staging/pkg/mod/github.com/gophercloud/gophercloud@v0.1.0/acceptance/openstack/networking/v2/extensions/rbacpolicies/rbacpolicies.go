package rbacpolicies

import (
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/rbacpolicies"
	th "github.com/gophercloud/gophercloud/testhelper"
)

// CreateRBACPolicy will create a rbac-policy. An error will be returned if the
// rbac-policy could not be created.
func CreateRBACPolicy(t *testing.T, client *gophercloud.ServiceClient, tenantID, networkID string) (*rbacpolicies.RBACPolicy, error) {
	createOpts := rbacpolicies.CreateOpts{
		Action:       rbacpolicies.ActionAccessShared,
		ObjectType:   "network",
		TargetTenant: tenantID,
		ObjectID:     networkID,
	}

	t.Logf("Trying to create rbac_policy")

	rbacPolicy, err := rbacpolicies.Create(client, createOpts).Extract()
	if err != nil {
		return rbacPolicy, err
	}

	t.Logf("Successfully created rbac_policy")

	th.AssertEquals(t, rbacPolicy.ObjectID, networkID)

	return rbacPolicy, nil
}

// DeleteRBACPolicy will delete a rbac-policy with a specified ID. A fatal error will
// occur if the delete was not successful. This works best when used as a
// deferred function.
func DeleteRBACPolicy(t *testing.T, client *gophercloud.ServiceClient, rbacPolicyID string) {
	t.Logf("Trying to delete rbac_policy: %s", rbacPolicyID)

	err := rbacpolicies.Delete(client, rbacPolicyID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete rbac_policy %s: %v", rbacPolicyID, err)
	}

	t.Logf("Deleted rbac_policy: %s", rbacPolicyID)
}
