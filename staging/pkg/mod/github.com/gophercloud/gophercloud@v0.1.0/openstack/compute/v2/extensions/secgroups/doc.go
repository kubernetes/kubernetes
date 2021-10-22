/*
Package secgroups provides the ability to manage security groups through the
Nova API.

This API has been deprecated and will be removed from a future release of the
Nova API service.

For environments that support this extension, this package can be used
regardless of if either Neutron or nova-network is used as the cloud's network
service.

Example to List Security Groups

	allPages, err := secroups.List(computeClient).AllPages()
	if err != nil {
		panic(err)
	}

	allSecurityGroups, err := secgroups.ExtractSecurityGroups(allPages)
	if err != nil {
		panic(err)
	}

	for _, sg := range allSecurityGroups {
		fmt.Printf("%+v\n", sg)
	}

Example to List Security Groups by Server

	serverID := "aab3ad01-9956-4623-a29b-24afc89a7d36"

	allPages, err := secroups.ListByServer(computeClient, serverID).AllPages()
	if err != nil {
		panic(err)
	}

	allSecurityGroups, err := secgroups.ExtractSecurityGroups(allPages)
	if err != nil {
		panic(err)
	}

	for _, sg := range allSecurityGroups {
		fmt.Printf("%+v\n", sg)
	}

Example to Create a Security Group

	createOpts := secgroups.CreateOpts{
		Name:        "group_name",
		Description: "A Security Group",
	}

	sg, err := secgroups.Create(computeClient, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Create a Security Group Rule

	sgID := "37d94f8a-d136-465c-ae46-144f0d8ef141"

	createOpts := secgroups.CreateRuleOpts{
		ParentGroupID: sgID,
		FromPort:      22,
		ToPort:        22,
		IPProtocol:    "tcp",
		CIDR:          "0.0.0.0/0",
	}

	rule, err := secgroups.CreateRule(computeClient, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Add a Security Group to a Server

	serverID := "aab3ad01-9956-4623-a29b-24afc89a7d36"
	sgID := "37d94f8a-d136-465c-ae46-144f0d8ef141"

	err := secgroups.AddServer(computeClient, serverID, sgID).ExtractErr()
	if err != nil {
		panic(err)
	}

Example to Remove a Security Group from a Server

	serverID := "aab3ad01-9956-4623-a29b-24afc89a7d36"
	sgID := "37d94f8a-d136-465c-ae46-144f0d8ef141"

	err := secgroups.RemoveServer(computeClient, serverID, sgID).ExtractErr()
	if err != nil {
		panic(err)
	}

Example to Delete a Security Group


	sgID := "37d94f8a-d136-465c-ae46-144f0d8ef141"
	err := secgroups.Delete(computeClient, sgID).ExtractErr()
	if err != nil {
		panic(err)
	}

Example to Delete a Security Group Rule

	ruleID := "6221fe3e-383d-46c9-a3a6-845e66c1e8b4"
	err := secgroups.DeleteRule(computeClient, ruleID).ExtractErr()
	if err != nil {
		panic(err)
	}
*/
package secgroups
