/*
Package external provides information and interaction with the external
extension for the OpenStack Networking service.

Example to List Networks with External Information

	type NetworkWithExternalExt struct {
		networks.Network
		external.NetworkExternalExt
	}

	var allNetworks []NetworkWithExternalExt

	allPages, err := networks.List(networkClient, nil).AllPages()
	if err != nil {
		panic(err)
	}

	err = networks.ExtractNetworksInto(allPages, &allNetworks)
	if err != nil {
		panic(err)
	}

	for _, network := range allNetworks {
		fmt.Println("%+v\n", network)
	}

Example to Create a Network with External Information

	iTrue := true
	networkCreateOpts := networks.CreateOpts{
		Name:         "private",
		AdminStateUp: &iTrue,
	}

	createOpts := external.CreateOptsExt{
		networkCreateOpts,
		&iTrue,
	}

	network, err := networks.Create(networkClient, createOpts).Extract()
	if err != nil {
		panic(err)
	}
*/
package external
