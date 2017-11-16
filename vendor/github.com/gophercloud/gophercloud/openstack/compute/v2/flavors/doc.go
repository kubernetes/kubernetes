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
*/
package flavors
