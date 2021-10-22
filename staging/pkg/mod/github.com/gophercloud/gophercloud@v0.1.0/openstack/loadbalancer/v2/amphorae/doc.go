/*
Package amphorae provides information and interaction with Amphorae
of OpenStack Load-balancing service.

Example to List Amphorae

	listOpts := amphorae.ListOpts{
		LoadbalancerID: "6bd55cd3-802e-447e-a518-1e74e23bb106",
	}

	allPages, err := amphorae.List(octaviaClient, listOpts).AllPages()
	if err != nil {
		panic(err)
	}

	allAmphorae, err := amphorae.ExtractAmphorae(allPages)
	if err != nil {
		panic(err)
	}

	for _, amphora := range allAmphorae {
		fmt.Printf("%+v\n", amphora)
	}
*/
package amphorae
