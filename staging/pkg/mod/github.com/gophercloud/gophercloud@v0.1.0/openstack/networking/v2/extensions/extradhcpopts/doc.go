/*
Package extradhcpopts allow to work with extra DHCP functionality of Neutron ports.

Example to Get a Port with Extra DHCP Options

	portID := "46d4bfb9-b26e-41f3-bd2e-e6dcc1ccedb2"
	var s struct {
		ports.Port
		extradhcpopts.ExtraDHCPOptsExt
	}

	err := ports.Get(networkClient, portID).ExtractInto(&s)
	if err != nil {
		panic(err)
	}

Example to Create a Port with Extra DHCP Options

	var s struct {
		ports.Port
		extradhcpopts.ExtraDHCPOptsExt
	}

	adminStateUp := true
	portCreateOpts := ports.CreateOpts{
		Name:         "dhcp-conf-port",
		AdminStateUp: &adminStateUp,
		NetworkID:    "a87cc70a-3e15-4acf-8205-9b711a3531b7",
		FixedIPs: []ports.IP{
			{SubnetID: "a0304c3a-4f08-4c43-88af-d796509c97d2", IPAddress: "10.0.0.2"},
		},
	}

	createOpts := extradhcpopts.CreateOptsExt{
		CreateOptsBuilder: portCreateOpts,
		ExtraDHCPOpts: []extradhcpopts.CreateExtraDHCPOpt{
			{
				OptName:  "optionA",
				OptValue: "valueA",
			},
		},
	}

	err := ports.Create(networkClient, createOpts).ExtractInto(&s)
	if err != nil {
		panic(err)
	}

Example to Update a Port with Extra DHCP Options

	var s struct {
		ports.Port
		extradhcpopts.ExtraDHCPOptsExt
	}

	portUpdateOpts := ports.UpdateOpts{
		Name: "updated-dhcp-conf-port",
		FixedIPs: []ports.IP{
			{SubnetID: "a0304c3a-4f08-4c43-88af-d796509c97d2", IPAddress: "10.0.0.3"},
		},
	}

	value := "valueB"
	updateOpts := extradhcpopts.UpdateOptsExt{
		UpdateOptsBuilder: portUpdateOpts,
		ExtraDHCPOpts: []extradhcpopts.UpdateExtraDHCPOpt{
			{
				OptName:  "optionB",
				OptValue: &value,
			},
		},
	}

	portID := "46d4bfb9-b26e-41f3-bd2e-e6dcc1ccedb2"
	err := ports.Update(networkClient, portID, updateOpts).ExtractInto(&s)
	if err != nil {
		panic(err)
	}
*/
package extradhcpopts
