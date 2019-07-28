/*
Package pools provides information and interaction with Pools and
Members of the LBaaS v2 extension for the OpenStack Networking service.

Example to List Pools

	listOpts := pools.ListOpts{
		LoadbalancerID: "c79a4468-d788-410c-bf79-9a8ef6354852",
	}

	allPages, err := pools.List(networkClient, listOpts).AllPages()
	if err != nil {
		panic(err)
	}

	allPools, err := pools.ExtractPools(allPages)
	if err != nil {
		panic(err)
	}

	for _, pools := range allPools {
		fmt.Printf("%+v\n", pool)
	}

Example to Create a Pool

	createOpts := pools.CreateOpts{
		LBMethod:       pools.LBMethodRoundRobin,
		Protocol:       "HTTP",
		Name:           "Example pool",
		LoadbalancerID: "79e05663-7f03-45d2-a092-8b94062f22ab",
	}

	pool, err := pools.Create(networkClient, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Update a Pool

	poolID := "d67d56a6-4a86-4688-a282-f46444705c64"

	updateOpts := pools.UpdateOpts{
		Name: "new-name",
	}

	pool, err := pools.Update(networkClient, poolID, updateOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Delete a Pool

	poolID := "d67d56a6-4a86-4688-a282-f46444705c64"
	err := pools.Delete(networkClient, poolID).ExtractErr()
	if err != nil {
		panic(err)
	}

Example to List Pool Members

	poolID := "d67d56a6-4a86-4688-a282-f46444705c64"

	listOpts := pools.ListMemberOpts{
		ProtocolPort: 80,
	}

	allPages, err := pools.ListMembers(networkClient, poolID, listOpts).AllPages()
	if err != nil {
		panic(err)
	}

	allMembers, err := pools.ExtractMembers(allPages)
	if err != nil {
		panic(err)
	}

	for _, member := allMembers {
		fmt.Printf("%+v\n", member)
	}

Example to Create a Member

	poolID := "d67d56a6-4a86-4688-a282-f46444705c64"

	weight := 10
	createOpts := pools.CreateMemberOpts{
		Name:         "db",
		SubnetID:     "1981f108-3c48-48d2-b908-30f7d28532c9",
		Address:      "10.0.2.11",
		ProtocolPort: 80,
		Weight:       &weight,
	}

	member, err := pools.CreateMember(networkClient, poolID, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Update a Member

	poolID := "d67d56a6-4a86-4688-a282-f46444705c64"
	memberID := "64dba99f-8af8-4200-8882-e32a0660f23e"

	weight := 4
	updateOpts := pools.UpdateMemberOpts{
		Name:   "new-name",
		Weight: &weight,
	}

	member, err := pools.UpdateMember(networkClient, poolID, memberID, updateOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Delete a Member

	poolID := "d67d56a6-4a86-4688-a282-f46444705c64"
	memberID := "64dba99f-8af8-4200-8882-e32a0660f23e"

	err := pools.DeleteMember(networkClient, poolID, memberID).ExtractErr()
	if err != nil {
		panic(err)
	}
*/
package pools
