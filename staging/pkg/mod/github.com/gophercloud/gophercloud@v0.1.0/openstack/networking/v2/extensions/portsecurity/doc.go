/*
Package portsecurity provides information and interaction with the port
security extension for the OpenStack Networking service.

Example to List Networks with Port Security Information

	type NetworkWithPortSecurityExt struct {
		networks.Network
		portsecurity.PortSecurityExt
	}

	var allNetworks []NetworkWithPortSecurityExt

	listOpts := networks.ListOpts{
		Name: "network_1",
	}

	allPages, err := networks.List(networkClient, listOpts).AllPages()
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

Example to Create a Network without Port Security

	var networkWithPortSecurityExt struct {
		networks.Network
		portsecurity.PortSecurityExt
	}

	networkCreateOpts := networks.CreateOpts{
		Name: "private",
	}

	iFalse := false
	createOpts := portsecurity.NetworkCreateOptsExt{
		CreateOptsBuilder:   networkCreateOpts,
		PortSecurityEnabled: &iFalse,
	}

	err := networks.Create(networkClient, createOpts).ExtractInto(&networkWithPortSecurityExt)
	if err != nil {
		panic(err)
	}

	fmt.Println("%+v\n", networkWithPortSecurityExt)

Example to Disable Port Security on an Existing Network

	var networkWithPortSecurityExt struct {
		networks.Network
		portsecurity.PortSecurityExt
	}

	iFalse := false
	networkID := "4e8e5957-649f-477b-9e5b-f1f75b21c03c"
	networkUpdateOpts := networks.UpdateOpts{}
	updateOpts := portsecurity.NetworkUpdateOptsExt{
		UpdateOptsBuilder:   networkUpdateOpts,
		PortSecurityEnabled: &iFalse,
	}

	err := networks.Update(networkClient, networkID, updateOpts).ExtractInto(&networkWithPortSecurityExt)
	if err != nil {
		panic(err)
	}

	fmt.Println("%+v\n", networkWithPortSecurityExt)

Example to Get a Port with Port Security Information

	var portWithPortSecurityExtensions struct {
		ports.Port
		portsecurity.PortSecurityExt
	}

	portID := "46d4bfb9-b26e-41f3-bd2e-e6dcc1ccedb2"

	err := ports.Get(networkingClient, portID).ExtractInto(&portWithPortSecurityExtensions)
	if err != nil {
		panic(err)
	}

	fmt.Println("%+v\n", portWithPortSecurityExtensions)

Example to Create a Port Without Port Security

	var portWithPortSecurityExtensions struct {
		ports.Port
		portsecurity.PortSecurityExt
	}

	iFalse := false
	networkID := "4e8e5957-649f-477b-9e5b-f1f75b21c03c"
	subnetID := "a87cc70a-3e15-4acf-8205-9b711a3531b7"

	portCreateOpts := ports.CreateOpts{
		NetworkID: networkID,
		FixedIPs:  []ports.IP{ports.IP{SubnetID: subnetID}},
	}

	createOpts := portsecurity.PortCreateOptsExt{
		CreateOptsBuilder:   portCreateOpts,
		PortSecurityEnabled: &iFalse,
	}

	err := ports.Create(networkingClient, createOpts).ExtractInto(&portWithPortSecurityExtensions)
	if err != nil {
		panic(err)
	}

	fmt.Println("%+v\n", portWithPortSecurityExtensions)

Example to Disable Port Security on an Existing Port

	var portWithPortSecurityExtensions struct {
		ports.Port
		portsecurity.PortSecurityExt
	}

	iFalse := false
	portID := "65c0ee9f-d634-4522-8954-51021b570b0d"

	portUpdateOpts := ports.UpdateOpts{}
	updateOpts := portsecurity.PortUpdateOptsExt{
		UpdateOptsBuilder:   portUpdateOpts,
		PortSecurityEnabled: &iFalse,
	}

	err := ports.Update(networkingClient, portID, updateOpts).ExtractInto(&portWithPortSecurityExtensions)
	if err != nil {
		panic(err)
	}

	fmt.Println("%+v\n", portWithPortSecurityExtensions)
*/
package portsecurity
