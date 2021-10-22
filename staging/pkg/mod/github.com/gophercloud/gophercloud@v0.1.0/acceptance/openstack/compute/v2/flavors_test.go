// +build acceptance compute flavors

package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	identity "github.com/gophercloud/gophercloud/acceptance/openstack/identity/v3"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/flavors"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestFlavorsList(t *testing.T) {
	client, err := clients.NewComputeV2Client()
	th.AssertNoErr(t, err)

	choices, err := clients.AcceptanceTestChoicesFromEnv()
	th.AssertNoErr(t, err)

	allPages, err := flavors.ListDetail(client, nil).AllPages()
	th.AssertNoErr(t, err)

	allFlavors, err := flavors.ExtractFlavors(allPages)
	th.AssertNoErr(t, err)

	var found bool
	for _, flavor := range allFlavors {
		tools.PrintResource(t, flavor)

		if flavor.ID == choices.FlavorID {
			found = true
		}
	}

	th.AssertEquals(t, found, true)
}

func TestFlavorsAccessTypeList(t *testing.T) {
	client, err := clients.NewComputeV2Client()
	th.AssertNoErr(t, err)

	flavorAccessTypes := map[string]flavors.AccessType{
		"public":  flavors.PublicAccess,
		"private": flavors.PrivateAccess,
		"all":     flavors.AllAccess,
	}

	for flavorTypeName, flavorAccessType := range flavorAccessTypes {
		t.Logf("** %s flavors: **", flavorTypeName)
		allPages, err := flavors.ListDetail(client, flavors.ListOpts{AccessType: flavorAccessType}).AllPages()
		th.AssertNoErr(t, err)

		allFlavors, err := flavors.ExtractFlavors(allPages)
		th.AssertNoErr(t, err)

		for _, flavor := range allFlavors {
			tools.PrintResource(t, flavor)
		}
	}
}

func TestFlavorsGet(t *testing.T) {
	client, err := clients.NewComputeV2Client()
	th.AssertNoErr(t, err)

	choices, err := clients.AcceptanceTestChoicesFromEnv()
	th.AssertNoErr(t, err)

	flavor, err := flavors.Get(client, choices.FlavorID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, flavor)

	th.AssertEquals(t, flavor.ID, choices.FlavorID)
}

func TestFlavorsCreateDelete(t *testing.T) {
	clients.RequireAdmin(t)

	client, err := clients.NewComputeV2Client()
	th.AssertNoErr(t, err)

	flavor, err := CreateFlavor(t, client)
	th.AssertNoErr(t, err)
	defer DeleteFlavor(t, client, flavor)

	tools.PrintResource(t, flavor)
}

func TestFlavorsAccessesList(t *testing.T) {
	clients.RequireAdmin(t)

	client, err := clients.NewComputeV2Client()
	th.AssertNoErr(t, err)

	flavor, err := CreatePrivateFlavor(t, client)
	th.AssertNoErr(t, err)
	defer DeleteFlavor(t, client, flavor)

	allPages, err := flavors.ListAccesses(client, flavor.ID).AllPages()
	th.AssertNoErr(t, err)

	allAccesses, err := flavors.ExtractAccesses(allPages)
	th.AssertNoErr(t, err)

	th.AssertEquals(t, len(allAccesses), 0)
}

func TestFlavorsAccessCRUD(t *testing.T) {
	clients.RequireAdmin(t)

	client, err := clients.NewComputeV2Client()
	th.AssertNoErr(t, err)

	identityClient, err := clients.NewIdentityV3Client()
	th.AssertNoErr(t, err)

	project, err := identity.CreateProject(t, identityClient, nil)
	th.AssertNoErr(t, err)
	defer identity.DeleteProject(t, identityClient, project.ID)

	flavor, err := CreatePrivateFlavor(t, client)
	th.AssertNoErr(t, err)
	defer DeleteFlavor(t, client, flavor)

	addAccessOpts := flavors.AddAccessOpts{
		Tenant: project.ID,
	}

	accessList, err := flavors.AddAccess(client, flavor.ID, addAccessOpts).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, len(accessList), 1)
	th.AssertEquals(t, accessList[0].TenantID, project.ID)
	th.AssertEquals(t, accessList[0].FlavorID, flavor.ID)

	for _, access := range accessList {
		tools.PrintResource(t, access)
	}

	removeAccessOpts := flavors.RemoveAccessOpts{
		Tenant: project.ID,
	}

	accessList, err = flavors.RemoveAccess(client, flavor.ID, removeAccessOpts).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, len(accessList), 0)
}

func TestFlavorsExtraSpecsCRUD(t *testing.T) {
	clients.RequireAdmin(t)

	client, err := clients.NewComputeV2Client()
	th.AssertNoErr(t, err)

	flavor, err := CreatePrivateFlavor(t, client)
	th.AssertNoErr(t, err)
	defer DeleteFlavor(t, client, flavor)

	createOpts := flavors.ExtraSpecsOpts{
		"hw:cpu_policy":        "CPU-POLICY",
		"hw:cpu_thread_policy": "CPU-THREAD-POLICY",
	}
	createdExtraSpecs, err := flavors.CreateExtraSpecs(client, flavor.ID, createOpts).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, createdExtraSpecs)

	th.AssertEquals(t, len(createdExtraSpecs), 2)
	th.AssertEquals(t, createdExtraSpecs["hw:cpu_policy"], "CPU-POLICY")
	th.AssertEquals(t, createdExtraSpecs["hw:cpu_thread_policy"], "CPU-THREAD-POLICY")

	err = flavors.DeleteExtraSpec(client, flavor.ID, "hw:cpu_policy").ExtractErr()
	th.AssertNoErr(t, err)

	updateOpts := flavors.ExtraSpecsOpts{
		"hw:cpu_thread_policy": "CPU-THREAD-POLICY-BETTER",
	}
	updatedExtraSpec, err := flavors.UpdateExtraSpec(client, flavor.ID, updateOpts).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, updatedExtraSpec)

	allExtraSpecs, err := flavors.ListExtraSpecs(client, flavor.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, allExtraSpecs)

	th.AssertEquals(t, len(allExtraSpecs), 1)
	th.AssertEquals(t, allExtraSpecs["hw:cpu_thread_policy"], "CPU-THREAD-POLICY-BETTER")

	spec, err := flavors.GetExtraSpec(client, flavor.ID, "hw:cpu_thread_policy").Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, spec)

	th.AssertEquals(t, spec["hw:cpu_thread_policy"], "CPU-THREAD-POLICY-BETTER")
}
