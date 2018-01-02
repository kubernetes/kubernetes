/*
Package pools provides information and interaction with the Pools of the
Load Balancing as a Service extension for the OpenStack Networking service.

Example to List Pools

	listOpts := pools.ListOpts{
		SubnetID: "d9bd223b-f1a9-4f98-953b-df977b0f902d",
	}

	allPages, err := pools.List(networkClient, listOpts).AllPages()
	if err != nil {
		panic(err)
	}

	allPools, err := pools.ExtractPools(allPages)
	if err != nil {
		panic(err)
	}

	for _, pool := range allPools {
		fmt.Printf("%+v\n", pool)
	}

Example to Create a Pool

	createOpts := pools.CreateOpts{
		LBMethod: pools.LBMethodRoundRobin,
		Protocol: "HTTP",
		Name:     "Example pool",
		SubnetID: "1981f108-3c48-48d2-b908-30f7d28532c9",
		Provider: "haproxy",
	}

	pool, err := pools.Create(networkClient, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Update a Pool

	poolID := "166db5e6-c72a-4d77-8776-3573e27ae271"

	updateOpts := pools.UpdateOpts{
		LBMethod: pools.LBMethodLeastConnections,
	}

	pool, err := pools.Update(networkClient, poolID, updateOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Delete a Pool

	poolID := "166db5e6-c72a-4d77-8776-3573e27ae271"
	err := pools.Delete(networkClient, poolID).ExtractErr()
	if err != nil {
		panic(err)
	}

Example to Associate a Monitor to a Pool

	poolID := "166db5e6-c72a-4d77-8776-3573e27ae271"
	monitorID := "8bbfbe1c-6faa-4d97-abdb-0df6c90df70b"

	pool, err := pools.AssociateMonitor(networkClient, poolID, monitorID).Extract()
	if err != nil {
		panic(err)
	}

Example to Disassociate a Monitor from a Pool

	poolID := "166db5e6-c72a-4d77-8776-3573e27ae271"
	monitorID := "8bbfbe1c-6faa-4d97-abdb-0df6c90df70b"

	pool, err := pools.DisassociateMonitor(networkClient, poolID, monitorID).Extract()
	if err != nil {
		panic(err)
	}
*/
package pools
