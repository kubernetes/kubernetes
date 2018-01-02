/*
Package members enables management and retrieval of image members.

Members are projects other than the image owner who have access to the image.

Example to List Members of an Image

	imageID := "2b6cacd4-cfd6-4b95-8302-4c04ccf0be3f"

	allPages, err := members.List(imageID).AllPages()
	if err != nil {
		panic(err)
	}

	allMembers, err := members.ExtractMembers(allPages)
	if err != nil {
		panic(err)
	}

	for _, member := range allMembers {
		fmt.Printf("%+v\n", member)
	}

Example to Add a Member to an Image

	imageID := "2b6cacd4-cfd6-4b95-8302-4c04ccf0be3f"
	projectID := "fc404778935a4cebaddcb4788fb3ff2c"

	member, err := members.Create(imageClient, imageID, projectID).Extract()
	if err != nil {
		panic(err)
	}

Example to Update the Status of a Member

	imageID := "2b6cacd4-cfd6-4b95-8302-4c04ccf0be3f"
	projectID := "fc404778935a4cebaddcb4788fb3ff2c"

	updateOpts := members.UpdateOpts{
		Status: "accepted",
	}

	member, err := members.Update(imageClient, imageID, projectID, updateOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Delete a Member from an Image

	imageID := "2b6cacd4-cfd6-4b95-8302-4c04ccf0be3f"
	projectID := "fc404778935a4cebaddcb4788fb3ff2c"

	err := members.Delete(imageClient, imageID, projectID).ExtractErr()
	if err != nil {
		panic(err)
	}
*/
package members
