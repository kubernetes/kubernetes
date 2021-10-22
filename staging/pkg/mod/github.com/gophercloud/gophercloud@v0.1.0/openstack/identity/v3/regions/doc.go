/*
Package regions manages and retrieves Regions in the OpenStack Identity Service.

Example to List Regions

	listOpts := regions.ListOpts{
		ParentRegionID: "RegionOne",
	}

	allPages, err := regions.List(identityClient, listOpts).AllPages()
	if err != nil {
		panic(err)
	}

	allRegions, err := regions.ExtractRegions(allPages)
	if err != nil {
		panic(err)
	}

	for _, region := range allRegions {
		fmt.Printf("%+v\n", region)
	}

Example to Create a Region

	createOpts := regions.CreateOpts{
		ID:             "TestRegion",
		Description: "Region for testing"
		Extra: map[string]interface{}{
			"email": "testregionsupport@example.com",
		}
	}

	region, err := regions.Create(identityClient, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Update a Region

	regionID := "TestRegion"

	// There is currently a bug in Keystone where updating the optional Extras
	// attributes set in regions.Create is not supported, see:
	// https://bugs.launchpad.net/keystone/+bug/1729933
	updateOpts := regions.UpdateOpts{
		Description: "Updated Description for region",
	}

	region, err := regions.Update(identityClient, regionID, updateOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Delete a Region

	regionID := "TestRegion"
	err := regions.Delete(identityClient, regionID).ExtractErr()
	if err != nil {
		panic(err)
	}
*/
package regions
