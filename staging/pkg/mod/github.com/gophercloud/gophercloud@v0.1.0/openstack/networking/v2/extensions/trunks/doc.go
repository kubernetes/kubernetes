/*
Package trunks provides the ability to retrieve and manage trunks through the Neutron API.
Trunks allow you to multiplex multiple ports traffic on a single port. For example, you could
have a compute instance port be the parent port of a trunk and inside the VM run workloads
using other ports, without the need of plugging those ports.

Example of a new empty Trunk creation

	iTrue := true
	createOpts := trunks.CreateOpts{
		Name:         "gophertrunk",
		Description:  "Trunk created by gophercloud",
		AdminStateUp: &iTrue,
		PortID:       "a6f0560c-b7a8-401f-bf6e-d0a5c851ae10",
	}

	trunk, err := trunks.Create(networkClient, trunkOpts).Extract()
	if err != nil {
		panic(err)
	}
	fmt.Printf("%+v\n", trunk)

Example of a new Trunk creation with 2 subports

	iTrue := true
	createOpts := trunks.CreateOpts{
		Name:         "gophertrunk",
		Description:  "Trunk created by gophercloud",
		AdminStateUp: &iTrue,
		PortID:       "a6f0560c-b7a8-401f-bf6e-d0a5c851ae10",
		Subports: []trunks.Subport{
			{
				SegmentationID:   1,
				SegmentationType: "vlan",
				PortID:           "bf4efcc0-b1c7-4674-81f0-31f58a33420a",
			},
			{
				SegmentationID:   10,
				SegmentationType: "vlan",
				PortID:           "2cf671b9-02b3-4121-9e85-e0af3548d112",
			},
		},
	}

	trunk, err := trunks.Create(client, trunkOpts).Extract()
	if err != nil {
		panic(err)
	}
	fmt.Printf("%+v\n", trunk)

Example of deleting a Trunk

	trunkID := "c36e7f2e-0c53-4742-8696-aee77c9df159"
	err := trunks.Delete(networkClient, trunkID).ExtractErr()
	if err != nil {
		panic(err)
	}

Example of listing Trunks

	listOpts := trunks.ListOpts{}
	allPages, err := trunks.List(networkClient, listOpts).AllPages()
	if err != nil {
		panic(err)
	}
	allTrunks, err := trunks.ExtractTrunks(allPages)
	if err != nil {
		panic(err)
	}
	for _, trunk := range allTrunks {
		fmt.Printf("%+v\n", trunk)
	}

Example of getting a Trunk

	trunkID = "52d8d124-3dc9-4563-9fef-bad3187ecf2d"
	trunk, err := trunks.Get(networkClient, trunkID).Extract()
	if err != nil {
		panic(err)
	}
	fmt.Printf("%+v\n", trunk)

Example of updating a Trunk

	trunkID := "c36e7f2e-0c53-4742-8696-aee77c9df159"
	subports, err := trunks.GetSubports(client, trunkID).Extract()
	iFalse := false
	updateOpts := trunks.UpdateOpts{
		AdminStateUp: &iFalse,
		Name:         "updated_gophertrunk",
		Description:  "trunk updated by gophercloud",
	}
	trunk, err = trunks.Update(networkClient, trunkID, updateOpts).Extract()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%+v\n", trunk)

Example of showing subports of a Trunk

	trunkID := "c36e7f2e-0c53-4742-8696-aee77c9df159"
	subports, err := trunks.GetSubports(client, trunkID).Extract()
	fmt.Printf("%+v\n", subports)

Example of adding two subports to a Trunk

	trunkID := "c36e7f2e-0c53-4742-8696-aee77c9df159"
	addSubportsOpts := trunks.AddSubportsOpts{
		Subports: []trunks.Subport{
			{
				SegmentationID:   1,
				SegmentationType: "vlan",
				PortID:           "bf4efcc0-b1c7-4674-81f0-31f58a33420a",
			},
			{
				SegmentationID:   10,
				SegmentationType: "vlan",
				PortID:           "2cf671b9-02b3-4121-9e85-e0af3548d112",
			},
		},
	}
	trunk, err := trunks.AddSubports(client, trunkID, addSubportsOpts).Extract()
	if err != nil {
		panic(err)
	}
	fmt.Printf("%+v\n", trunk)

Example of deleting two subports from a Trunk

	trunkID := "c36e7f2e-0c53-4742-8696-aee77c9df159"
	removeSubportsOpts := trunks.RemoveSubportsOpts{
		Subports: []trunks.RemoveSubport{
			{PortID: "bf4efcc0-b1c7-4674-81f0-31f58a33420a"},
			{PortID: "2cf671b9-02b3-4121-9e85-e0af3548d112"},
		},
	}
	trunk, err := trunks.RemoveSubports(networkClient, trunkID, removeSubportsOpts).Extract()
	if err != nil {
		panic(err)
	}
	fmt.Printf("%+v\n", trunk)
*/
package trunks
