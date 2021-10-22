/*
Package projects manages and retrieves Projects in the OpenStack Identity
Service.

Example to List Projects

	listOpts := projects.ListOpts{
		Enabled: gophercloud.Enabled,
	}

	allPages, err := projects.List(identityClient, listOpts).AllPages()
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

Example to Create a Project

	createOpts := projects.CreateOpts{
		Name:        "project_name",
		Description: "Project Description"
	}

	project, err := projects.Create(identityClient, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Update a Project

	projectID := "966b3c7d36a24facaf20b7e458bf2192"

	updateOpts := projects.UpdateOpts{
		Enabled: gophercloud.Disabled,
	}

	project, err := projects.Update(identityClient, projectID, updateOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Delete a Project

	projectID := "966b3c7d36a24facaf20b7e458bf2192"
	err := projects.Delete(identityClient, projectID).ExtractErr()
	if err != nil {
		panic(err)
	}
*/
package projects
