// +build acceptance compute flavors

package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/flavors"
)

func TestFlavorsList(t *testing.T) {
	t.Logf("** Default flavors (same as Project flavors): **")
	t.Logf("")
	client, err := clients.NewComputeV2Client()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	allPages, err := flavors.ListDetail(client, nil).AllPages()
	if err != nil {
		t.Fatalf("Unable to retrieve flavors: %v", err)
	}

	allFlavors, err := flavors.ExtractFlavors(allPages)
	if err != nil {
		t.Fatalf("Unable to extract flavor results: %v", err)
	}

	for _, flavor := range allFlavors {
		tools.PrintResource(t, flavor)
	}

	flavorAccessTypes := [3]flavors.AccessType{flavors.PublicAccess, flavors.PrivateAccess, flavors.AllAccess}
	for _, flavorAccessType := range flavorAccessTypes {
		t.Logf("** %s flavors: **", flavorAccessType)
		t.Logf("")
		allPages, err := flavors.ListDetail(client, flavors.ListOpts{AccessType: flavorAccessType}).AllPages()
		if err != nil {
			t.Fatalf("Unable to retrieve flavors: %v", err)
		}

		allFlavors, err := flavors.ExtractFlavors(allPages)
		if err != nil {
			t.Fatalf("Unable to extract flavor results: %v", err)
		}

		for _, flavor := range allFlavors {
			tools.PrintResource(t, flavor)
			t.Logf("")
		}
	}

}

func TestFlavorsGet(t *testing.T) {
	client, err := clients.NewComputeV2Client()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	choices, err := clients.AcceptanceTestChoicesFromEnv()
	if err != nil {
		t.Fatal(err)
	}

	flavor, err := flavors.Get(client, choices.FlavorID).Extract()
	if err != nil {
		t.Fatalf("Unable to get flavor information: %v", err)
	}

	tools.PrintResource(t, flavor)
}

func TestFlavorCreateDelete(t *testing.T) {
	client, err := clients.NewComputeV2Client()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	flavor, err := CreateFlavor(t, client)
	if err != nil {
		t.Fatalf("Unable to create flavor: %v", err)
	}
	defer DeleteFlavor(t, client, flavor)

	tools.PrintResource(t, flavor)
}

func TestFlavorAccessesList(t *testing.T) {
	client, err := clients.NewComputeV2Client()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	flavor, err := CreatePrivateFlavor(t, client)
	if err != nil {
		t.Fatalf("Unable to create flavor: %v", err)
	}
	defer DeleteFlavor(t, client, flavor)

	allPages, err := flavors.ListAccesses(client, flavor.ID).AllPages()
	if err != nil {
		t.Fatalf("Unable to list flavor accesses: %v", err)
	}

	allAccesses, err := flavors.ExtractAccesses(allPages)
	if err != nil {
		t.Fatalf("Unable to extract accesses: %v", err)
	}

	for _, access := range allAccesses {
		tools.PrintResource(t, access)
	}
}
