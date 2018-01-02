/*
Package users provides information and interaction with the users API
resource for the OpenStack Identity Service.

Example to List Users

	allPages, err := users.List(identityClient).AllPages()
	if err != nil {
		panic(err)
	}

	allUsers, err := users.ExtractUsers(allPages)
	if err != nil {
		panic(err)
	}

	for _, user := range allUsers {
		fmt.Printf("%+v\n", user)
	}

Example to Create a User

	createOpts := users.CreateOpts{
		Name:     "name",
		TenantID: "c39e3de9be2d4c779f1dfd6abacc176d",
		Enabled:  gophercloud.Enabled,
	}

	user, err := users.Create(identityClient, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Update a User

	userID := "9fe2ff9ee4384b1894a90878d3e92bab"

	updateOpts := users.UpdateOpts{
		Name:    "new_name",
		Enabled: gophercloud.Disabled,
	}

	user, err := users.Update(identityClient, userID, updateOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Delete a User

	userID := "9fe2ff9ee4384b1894a90878d3e92bab"
	err := users.Delete(identityClient, userID).ExtractErr()
	if err != nil {
		panic(err)
	}

Example to List a User's Roles

	tenantID := "1d8b6120dcc640fda4fc9194ffc80273"
	userID := "c39e3de9be2d4c779f1dfd6abacc176d"

	allPages, err := users.ListRoles(identityClient, tenantID, userID).AllPages()
	if err != nil {
		panic(err)
	}

	allRoles, err := users.ExtractRoles(allPages)
	if err != nil {
		panic(err)
	}

	for _, role := range allRoles {
		fmt.Printf("%+v\n", role)
	}
*/
package users
