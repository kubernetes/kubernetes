/*
Package users manages and retrieves Users in the OpenStack Identity Service.

Example to List Users

	listOpts := users.ListOpts{
		DomainID: "default",
	}

	allPages, err := users.List(identityClient, listOpts).AllPages()
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

	projectID := "a99e9b4e620e4db09a2dfb6e42a01e66"

	createOpts := users.CreateOpts{
		Name:             "username",
		DomainID:         "default",
		DefaultProjectID: projectID,
		Enabled:          gophercloud.Enabled,
		Password:         "supersecret",
		Extra: map[string]interface{}{
			"email": "username@example.com",
		}
	}

	user, err := users.Create(identityClient, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Update a User

	userID := "0fe36e73809d46aeae6705c39077b1b3"

	updateOpts := users.UpdateOpts{
		Enabled: gophercloud.Disabled,
	}

	user, err := users.Update(identityClient, userID, updateOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Change Password of a User

	userID := "0fe36e73809d46aeae6705c39077b1b3"
	originalPassword := "secretsecret"
	password := "new_secretsecret"

	changePasswordOpts := users.ChangePasswordOpts{
		OriginalPassword: originalPassword,
		Password:         password,
	}

	err := users.ChangePassword(identityClient, userID, changePasswordOpts).ExtractErr()
	if err != nil {
		panic(err)
	}

Example to Delete a User

	userID := "0fe36e73809d46aeae6705c39077b1b3"
	err := users.Delete(identityClient, userID).ExtractErr()
	if err != nil {
		panic(err)
	}

Example to List Groups a User Belongs To

	userID := "0fe36e73809d46aeae6705c39077b1b3"

	allPages, err := users.ListGroups(identityClient, userID).AllPages()
	if err != nil {
		panic(err)
	}

	allGroups, err := groups.ExtractGroups(allPages)
	if err != nil {
		panic(err)
	}

	for _, group := range allGroups {
		fmt.Printf("%+v\n", group)
	}

Example to Add a User to a Group

	groupID := "bede500ee1124ae9b0006ff859758b3a"
	userID := "0fe36e73809d46aeae6705c39077b1b3"
	err := users.AddToGroup(identityClient, groupID, userID).ExtractErr()

	if err != nil {
		panic(err)
	}

Example to Check Whether a User Belongs to a Group

	groupID := "bede500ee1124ae9b0006ff859758b3a"
	userID := "0fe36e73809d46aeae6705c39077b1b3"
	ok, err := users.IsMemberOfGroup(identityClient, groupID, userID).Extract()
	if err != nil {
		panic(err)
	}

	if ok {
		fmt.Printf("user %s is a member of group %s\n", userID, groupID)
	}

Example to Remove a User from a Group

	groupID := "bede500ee1124ae9b0006ff859758b3a"
	userID := "0fe36e73809d46aeae6705c39077b1b3"
	err := users.RemoveFromGroup(identityClient, groupID, userID).ExtractErr()

	if err != nil {
		panic(err)
	}

Example to List Projects a User Belongs To

	userID := "0fe36e73809d46aeae6705c39077b1b3"

	allPages, err := users.ListProjects(identityClient, userID).AllPages()
	if err != nil {
		panic(err)
	}

	allProjects, err := projects.ExtractProjects(allPages)
	if err != nil {
		panic(err)
	}

	for _, project := range allProjects {
		fmt.Printf("%+v\n", project)
	}

Example to List Users in a Group

	groupID := "bede500ee1124ae9b0006ff859758b3a"
	listOpts := users.ListOpts{
		DomainID: "default",
	}

	allPages, err := users.ListInGroup(identityClient, groupID, listOpts).AllPages()
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

*/
package users
