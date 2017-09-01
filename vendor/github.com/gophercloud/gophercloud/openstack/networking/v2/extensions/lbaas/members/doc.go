/*
Package members provides information and interaction with Members of the
Load Balancer as a Service extension for the OpenStack Networking service.

Example to List Members

	listOpts := members.ListOpts{
		ProtocolPort: 80,
	}

	allPages, err := members.List(networkClient, listOpts).AllPages()
	if err != nil {
		panic(err)
	}

	allMembers, err := members.ExtractMembers(allPages)
	if err != nil {
		panic(err)
	}

	for _, member := range allMembers {
		fmt.Printf("%+v\n", member)
	}

Example to Create a Member

	createOpts := members.CreateOpts{
		Address:      "192.168.2.14",
		ProtocolPort: 80,
		PoolID:       "0b266a12-0fdf-4434-bd11-649d84e54bd5"
	}

	member, err := members.Create(networkClient, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Update a Member

	memberID := "46592c54-03f7-40ef-9cdf-b1fcf2775ddf"

	updateOpts := members.UpdateOpts{
		AdminStateUp: gophercloud.Disabled,
	}

	member, err := members.Update(networkClient, memberID, updateOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Delete a Member

	memberID := "46592c54-03f7-40ef-9cdf-b1fcf2775ddf"
	err := members.Delete(networkClient, memberID).ExtractErr()
	if err != nil {
		panic(err)
	}
*/
package members
