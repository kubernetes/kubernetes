/*
Package servergroups provides the ability to manage server groups.

Example to List Server Groups

	allpages, err := servergroups.List(computeClient).AllPages()
	if err != nil {
		panic(err)
	}

	allServerGroups, err := servergroups.ExtractServerGroups(allPages)
	if err != nil {
		panic(err)
	}

	for _, sg := range allServerGroups {
		fmt.Printf("%#v\n", sg)
	}

Example to Create a Server Group

	createOpts := servergroups.CreateOpts{
		Name:     "my_sg",
		Policies: []string{"anti-affinity"},
	}

	sg, err := servergroups.Create(computeClient, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Delete a Server Group

	sgID := "7a6f29ad-e34d-4368-951a-58a08f11cfb7"
	err := servergroups.Delete(computeClient, sgID).ExtractErr()
	if err != nil {
		panic(err)
	}
*/
package servergroups
