/*
Package roles provides information and interaction with the roles API
resource for the OpenStack Identity service.

Example to List Roles

	listOpts := roles.ListOpts{
		DomainID: "default",
	}

	allPages, err := roles.List(identityClient, listOpts).AllPages()
	if err != nil {
		panic(err)
	}

	allRoles, err := roles.ExtractRoles(allPages)
	if err != nil {
		panic(err)
	}

	for _, role := range allRoles {
		fmt.Printf("%+v\n", role)
	}

Example to Create a Role

	createOpts := roles.CreateOpts{
		Name:             "read-only-admin",
		DomainID:         "default",
		Extra: map[string]interface{}{
			"description": "this role grants read-only privilege cross tenant",
		}
	}

	role, err := roles.Create(identityClient, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Update a Role

	roleID := "0fe36e73809d46aeae6705c39077b1b3"

	updateOpts := roles.UpdateOpts{
		Name: "read only admin",
	}

	role, err := roles.Update(identityClient, roleID, updateOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Delete a Role

	roleID := "0fe36e73809d46aeae6705c39077b1b3"
	err := roles.Delete(identityClient, roleID).ExtractErr()
	if err != nil {
		panic(err)
	}

Example to List Role Assignments

	listOpts := roles.ListAssignmentsOpts{
		UserID:         "97061de2ed0647b28a393c36ab584f39",
		ScopeProjectID: "9df1a02f5eb2416a9781e8b0c022d3ae",
	}

	allPages, err := roles.ListAssignments(identityClient, listOpts).AllPages()
	if err != nil {
		panic(err)
	}

	allRoles, err := roles.ExtractRoleAssignments(allPages)
	if err != nil {
		panic(err)
	}

	for _, role := range allRoles {
		fmt.Printf("%+v\n", role)
	}

Example to List Role Assignments for a User on a Project

	projectID := "a99e9b4e620e4db09a2dfb6e42a01e66"
	userID := "9df1a02f5eb2416a9781e8b0c022d3ae"
	listAssignmentsOnResourceOpts := roles.ListAssignmentsOnResourceOpts{
		UserID:    userID,
		ProjectID: projectID,
	}

	allPages, err := roles.ListAssignmentsOnResource(identityClient, listAssignmentsOnResourceOpts).AllPages()
	if err != nil {
		panic(err)
	}

	allRoles, err := roles.ExtractRoles(allPages)
	if err != nil {
		panic(err)
	}

	for _, role := range allRoles {
		fmt.Printf("%+v\n", role)
	}

Example to Assign a Role to a User in a Project

	projectID := "a99e9b4e620e4db09a2dfb6e42a01e66"
	userID := "9df1a02f5eb2416a9781e8b0c022d3ae"
	roleID := "9fe2ff9ee4384b1894a90878d3e92bab"

	err := roles.Assign(identityClient, roleID, roles.AssignOpts{
		UserID:    userID,
		ProjectID: projectID,
	}).ExtractErr()

	if err != nil {
		panic(err)
	}

Example to Unassign a Role From a User in a Project

	projectID := "a99e9b4e620e4db09a2dfb6e42a01e66"
	userID := "9df1a02f5eb2416a9781e8b0c022d3ae"
	roleID := "9fe2ff9ee4384b1894a90878d3e92bab"

	err := roles.Unassign(identityClient, roleID, roles.UnassignOpts{
		UserID:    userID,
		ProjectID: projectID,
	}).ExtractErr()

	if err != nil {
		panic(err)
	}
*/
package roles
