/*
Package flavors provides information and interaction with the flavor API
in the OpenStack Compute service.

A flavor is an available hardware configuration for a server. Each flavor
has a unique combination of disk space, memory capacity and priority for CPU
time.

Example to List Flavors

	listOpts := flavors.ListOpts{
		AccessType: flavors.PublicAccess,
	}

	allPages, err := flavors.ListDetail(computeClient, listOpts).AllPages()
	if err != nil {
		panic(err)
	}

	allFlavors, err := flavors.ExtractFlavors(allPages)
	if err != nil {
		panic(err)
	}

	for _, flavor := range allFlavors {
		fmt.Printf("%+v\n", flavor)
	}

Example to Create a Flavor

	createOpts := flavors.CreateOpts{
		ID:         "1",
		Name:       "m1.tiny",
		Disk:       gophercloud.IntToPointer(1),
		RAM:        512,
		VCPUs:      1,
		RxTxFactor: 1.0,
	}

	flavor, err := flavors.Create(computeClient, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to List Flavor Access

	flavorID := "e91758d6-a54a-4778-ad72-0c73a1cb695b"

	allPages, err := flavors.ListAccesses(computeClient, flavorID).AllPages()
	if err != nil {
		panic(err)
	}

	allAccesses, err := flavors.ExtractAccesses(allPages)
	if err != nil {
		panic(err)
	}

	for _, access := range allAccesses {
		fmt.Printf("%+v", access)
	}

Example to Grant Access to a Flavor

	flavorID := "e91758d6-a54a-4778-ad72-0c73a1cb695b"

	accessOpts := flavors.AddAccessOpts{
		Tenant: "15153a0979884b59b0592248ef947921",
	}

	accessList, err := flavors.AddAccess(computeClient, flavor.ID, accessOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Remove/Revoke Access to a Flavor

	flavorID := "e91758d6-a54a-4778-ad72-0c73a1cb695b"

	accessOpts := flavors.RemoveAccessOpts{
		Tenant: "15153a0979884b59b0592248ef947921",
	}

	accessList, err := flavors.RemoveAccess(computeClient, flavor.ID, accessOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Create Extra Specs for a Flavor

	flavorID := "e91758d6-a54a-4778-ad72-0c73a1cb695b"

	createOpts := flavors.ExtraSpecsOpts{
		"hw:cpu_policy":        "CPU-POLICY",
		"hw:cpu_thread_policy": "CPU-THREAD-POLICY",
	}
	createdExtraSpecs, err := flavors.CreateExtraSpecs(computeClient, flavorID, createOpts).Extract()
	if err != nil {
		panic(err)
	}

	fmt.Printf("%+v", createdExtraSpecs)

Example to Get Extra Specs for a Flavor

	flavorID := "e91758d6-a54a-4778-ad72-0c73a1cb695b"

	extraSpecs, err := flavors.ListExtraSpecs(computeClient, flavorID).Extract()
	if err != nil {
		panic(err)
	}

	fmt.Printf("%+v", extraSpecs)

Example to Update Extra Specs for a Flavor

	flavorID := "e91758d6-a54a-4778-ad72-0c73a1cb695b"

	updateOpts := flavors.ExtraSpecsOpts{
		"hw:cpu_thread_policy": "CPU-THREAD-POLICY-UPDATED",
	}
	updatedExtraSpec, err := flavors.UpdateExtraSpec(computeClient, flavorID, updateOpts).Extract()
	if err != nil {
		panic(err)
	}

	fmt.Printf("%+v", updatedExtraSpec)

Example to Delete an Extra Spec for a Flavor

	flavorID := "e91758d6-a54a-4778-ad72-0c73a1cb695b"
	err := flavors.DeleteExtraSpec(computeClient, flavorID, "hw:cpu_thread_policy").ExtractErr()
	if err != nil {
		panic(err)
	}
*/
package flavors
