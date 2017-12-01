/*
Package attachinterfaces provides the ability to retrieve and manage network
interfaces through Nova.

Example of Listing a Server's Interfaces

	serverID := "b07e7a3b-d951-4efc-a4f9-ac9f001afb7f"
	allPages, err := attachinterfaces.List(computeClient, serverID).AllPages()
	if err != nil {
		panic(err)
	}

	allInterfaces, err := attachinterfaces.ExtractInterfaces(allPages)
	if err != nil {
		panic(err)
	}

	for _, interface := range allInterfaces {
		fmt.Printf("%+v\n", interface)
	}

Example to Get a Server's Interface

	portID = "0dde1598-b374-474e-986f-5b8dd1df1d4e"
	serverID := "b07e7a3b-d951-4efc-a4f9-ac9f001afb7f"
	interface, err := attachinterfaces.Get(computeClient, serverID, portID).Extract()
	if err != nil {
		panic(err)
	}
*/
package attachinterfaces
