// +build acceptance

package rbacpolicies

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	projects "github.com/gophercloud/gophercloud/acceptance/openstack/identity/v3"
	networking "github.com/gophercloud/gophercloud/acceptance/openstack/networking/v2"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/rbacpolicies"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestRBACPolicyCRUD(t *testing.T) {
	clients.RequireAdmin(t)

	client, err := clients.NewNetworkV2Client()
	th.AssertNoErr(t, err)

	// Create a network
	network, err := networking.CreateNetwork(t, client)
	th.AssertNoErr(t, err)
	defer networking.DeleteNetwork(t, client, network.ID)

	tools.PrintResource(t, network)

	identityClient, err := clients.NewIdentityV3Client()
	th.AssertNoErr(t, err)

	// Create a project/tenant
	project, err := projects.CreateProject(t, identityClient, nil)
	th.AssertNoErr(t, err)
	defer projects.DeleteProject(t, identityClient, project.ID)

	tools.PrintResource(t, project)

	// Create a rbac-policy
	rbacPolicy, err := CreateRBACPolicy(t, client, project.ID, network.ID)
	th.AssertNoErr(t, err)
	defer DeleteRBACPolicy(t, client, rbacPolicy.ID)

	tools.PrintResource(t, rbacPolicy)

	// Create another project/tenant for rbac-update
	project2, err := projects.CreateProject(t, identityClient, nil)
	th.AssertNoErr(t, err)
	defer projects.DeleteProject(t, identityClient, project2.ID)

	tools.PrintResource(t, project2)

	// Update a rbac-policy
	updateOpts := rbacpolicies.UpdateOpts{
		TargetTenant: project2.ID,
	}

	_, err = rbacpolicies.Update(client, rbacPolicy.ID, updateOpts).Extract()
	th.AssertNoErr(t, err)

	// Get the rbac-policy by ID
	t.Logf("Get rbac_policy by ID")
	newrbacPolicy, err := rbacpolicies.Get(client, rbacPolicy.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newrbacPolicy)
}

func TestRBACPolicyList(t *testing.T) {
	clients.RequireAdmin(t)

	client, err := clients.NewNetworkV2Client()
	th.AssertNoErr(t, err)

	type rbacPolicy struct {
		rbacpolicies.RBACPolicy
	}

	var allRBACPolicies []rbacPolicy

	allPages, err := rbacpolicies.List(client, nil).AllPages()
	th.AssertNoErr(t, err)

	err = rbacpolicies.ExtractRBACPolicesInto(allPages, &allRBACPolicies)
	th.AssertNoErr(t, err)

	for _, rbacpolicy := range allRBACPolicies {
		tools.PrintResource(t, rbacpolicy)
	}
}
