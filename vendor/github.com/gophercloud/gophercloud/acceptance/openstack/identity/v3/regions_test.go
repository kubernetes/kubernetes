// +build acceptance

package v3

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/identity/v3/regions"
)

func TestRegionsList(t *testing.T) {
	client, err := clients.NewIdentityV3Client()
	if err != nil {
		t.Fatalf("Unable to obtain an identity client: %v", err)
	}

	listOpts := regions.ListOpts{
		ParentRegionID: "RegionOne",
	}

	allPages, err := regions.List(client, listOpts).AllPages()
	if err != nil {
		t.Fatalf("Unable to list regions: %v", err)
	}

	allRegions, err := regions.ExtractRegions(allPages)
	if err != nil {
		t.Fatalf("Unable to extract regions: %v", err)
	}

	for _, region := range allRegions {
		tools.PrintResource(t, region)
	}
}

func TestRegionsGet(t *testing.T) {
	client, err := clients.NewIdentityV3Client()
	if err != nil {
		t.Fatalf("Unable to obtain an identity client: %v", err)
	}

	allPages, err := regions.List(client, nil).AllPages()
	if err != nil {
		t.Fatalf("Unable to list regions: %v", err)
	}

	allRegions, err := regions.ExtractRegions(allPages)
	if err != nil {
		t.Fatalf("Unable to extract regions: %v", err)
	}

	region := allRegions[0]
	p, err := regions.Get(client, region.ID).Extract()
	if err != nil {
		t.Fatalf("Unable to get region: %v", err)
	}

	tools.PrintResource(t, p)
}

func TestRegionsCRUD(t *testing.T) {
	client, err := clients.NewIdentityV3Client()
	if err != nil {
		t.Fatalf("Unable to obtain an identity client: %v", err)
	}

	createOpts := regions.CreateOpts{
		ID:          "testregion",
		Description: "Region for testing",
		Extra: map[string]interface{}{
			"email": "testregion@example.com",
		},
	}

	// Create region in the default domain
	region, err := CreateRegion(t, client, &createOpts)
	if err != nil {
		t.Fatalf("Unable to create region: %v", err)
	}
	defer DeleteRegion(t, client, region.ID)

	tools.PrintResource(t, region)
	tools.PrintResource(t, region.Extra)

	updateOpts := regions.UpdateOpts{
		Description: "Region A for testing",
		/*
			// Due to a bug in Keystone, the Extra column of the Region table
			// is not updatable, see: https://bugs.launchpad.net/keystone/+bug/1729933
			// The following lines should be uncommented once the fix is merged.

			Extra: map[string]interface{}{
				"email": "testregionA@example.com",
			},
		*/
	}

	newRegion, err := regions.Update(client, region.ID, updateOpts).Extract()
	if err != nil {
		t.Fatalf("Unable to update region: %v", err)
	}

	tools.PrintResource(t, newRegion)
	tools.PrintResource(t, newRegion.Extra)
}
