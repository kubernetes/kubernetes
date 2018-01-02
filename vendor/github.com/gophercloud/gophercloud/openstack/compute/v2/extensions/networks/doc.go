/*
Package networks provides the ability to create and manage networks in cloud
environments using nova-network.

This package can also be used to retrieve network details of Neutron-based
networks.

Example to List Networks

	allPages, err := networks.List(computeClient).AllPages()
	if err != nil {
		panic(err)
	}

	allNetworks, err := networks.ExtractNetworks(allPages)
	if err != nil {
		panic(err)
	}

	for _, network := range allNetworks {
		fmt.Printf("%+v\n", network)
	}
*/
package networks
