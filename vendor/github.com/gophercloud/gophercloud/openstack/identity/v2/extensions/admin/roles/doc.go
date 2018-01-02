/*
Package roles provides functionality to interact with and control roles on
the API.

A role represents a personality that a user can assume when performing a
specific set of operations. If a role includes a set of rights and
privileges, a user assuming that role inherits those rights and privileges.

When a token is generated, the list of roles that user can assume is returned
back to them. Services that are being called by that user determine how they
interpret the set of roles a user has and to which operations or resources
each role grants access.

It is up to individual services such as Compute or Image to assign meaning
to these roles. As far as the Identity service is concerned, a role is an
arbitrary name assigned by the user.

Example to List Roles

	allPages, err := roles.List(identityClient).AllPages()
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

Example to Grant a Role to a User

	tenantID := "a99e9b4e620e4db09a2dfb6e42a01e66"
	userID := "9df1a02f5eb2416a9781e8b0c022d3ae"
	roleID := "9fe2ff9ee4384b1894a90878d3e92bab"

	err := roles.AddUser(identityClient, tenantID, userID, roleID).ExtractErr()
	if err != nil {
		panic(err)
	}

Example to Remove a Role from a User

	tenantID := "a99e9b4e620e4db09a2dfb6e42a01e66"
	userID := "9df1a02f5eb2416a9781e8b0c022d3ae"
	roleID := "9fe2ff9ee4384b1894a90878d3e92bab"

	err := roles.DeleteUser(identityClient, tenantID, userID, roleID).ExtractErr()
	if err != nil {
		panic(err)
	}
*/
package roles
