/*
Package rules enables management and retrieval of Firewall Rules in the
OpenStack Networking Service.

Example to List Rules

	listOpts := rules.ListOpts{
		Protocol: rules.ProtocolAny,
	}

	allPages, err := rules.List(networkClient, listOpts).AllPages()
	if err != nil {
		panic(err)
	}

	allRules, err := rules.ExtractRules(allPages)
	if err != nil {
		panic(err)
	}

	for _, rule := range allRules {
		fmt.Printf("%+v\n", rule)
	}

Example to Create a Rule

	createOpts := rules.CreateOpts{
		Action:               "allow",
		Protocol:             rules.ProtocolTCP,
		Description:          "ssh",
		DestinationPort:      22,
		DestinationIPAddress: "192.168.1.0/24",
	}

	rule, err := rules.Create(networkClient, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Update a Rule

	ruleID := "f03bd950-6c56-4f5e-a307-45967078f507"
	newPort := 80
	newDescription := "http"

	updateOpts := rules.UpdateOpts{
		Description: &newDescription,
		port:        &newPort,
	}

	rule, err := rules.Update(networkClient, ruleID, updateOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Delete a Rule

	ruleID := "f03bd950-6c56-4f5e-a307-45967078f507"
	err := rules.Delete(networkClient, ruleID).ExtractErr()
	if err != nil {
		panic(err)
	}
*/
package rules
