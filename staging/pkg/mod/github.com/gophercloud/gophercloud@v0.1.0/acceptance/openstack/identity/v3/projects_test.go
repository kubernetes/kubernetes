// +build acceptance

package v3

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/identity/v3/projects"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestProjectsList(t *testing.T) {
	clients.RequireAdmin(t)

	client, err := clients.NewIdentityV3Client()
	th.AssertNoErr(t, err)

	var iTrue bool = true
	listOpts := projects.ListOpts{
		Enabled: &iTrue,
	}

	allPages, err := projects.List(client, listOpts).AllPages()
	th.AssertNoErr(t, err)

	allProjects, err := projects.ExtractProjects(allPages)
	th.AssertNoErr(t, err)

	var found bool
	for _, project := range allProjects {
		tools.PrintResource(t, project)

		if project.Name == "admin" {
			found = true
		}
	}

	th.AssertEquals(t, found, true)

	listOpts.Filters = map[string]string{
		"name__contains": "dmi",
	}

	allPages, err = projects.List(client, listOpts).AllPages()
	th.AssertNoErr(t, err)

	allProjects, err = projects.ExtractProjects(allPages)
	th.AssertNoErr(t, err)

	found = false
	for _, project := range allProjects {
		tools.PrintResource(t, project)

		if project.Name == "admin" {
			found = true
		}
	}

	th.AssertEquals(t, found, true)

	listOpts.Filters = map[string]string{
		"name__contains": "foo",
	}

	allPages, err = projects.List(client, listOpts).AllPages()
	th.AssertNoErr(t, err)

	allProjects, err = projects.ExtractProjects(allPages)
	th.AssertNoErr(t, err)

	found = false
	for _, project := range allProjects {
		tools.PrintResource(t, project)

		if project.Name == "admin" {
			found = true
		}
	}

	th.AssertEquals(t, found, false)
}

func TestProjectsGet(t *testing.T) {
	clients.RequireAdmin(t)

	client, err := clients.NewIdentityV3Client()
	th.AssertNoErr(t, err)

	allPages, err := projects.List(client, nil).AllPages()
	th.AssertNoErr(t, err)

	allProjects, err := projects.ExtractProjects(allPages)
	th.AssertNoErr(t, err)

	project := allProjects[0]
	p, err := projects.Get(client, project.ID).Extract()
	if err != nil {
		t.Fatalf("Unable to get project: %v", err)
	}

	tools.PrintResource(t, p)

	th.AssertEquals(t, project.Name, p.Name)
}

func TestProjectsCRUD(t *testing.T) {
	clients.RequireAdmin(t)

	client, err := clients.NewIdentityV3Client()
	th.AssertNoErr(t, err)

	project, err := CreateProject(t, client, nil)
	th.AssertNoErr(t, err)
	defer DeleteProject(t, client, project.ID)

	tools.PrintResource(t, project)

	description := ""
	iFalse := false
	updateOpts := projects.UpdateOpts{
		Description: &description,
		Enabled:     &iFalse,
	}

	updatedProject, err := projects.Update(client, project.ID, updateOpts).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, updatedProject)
	th.AssertEquals(t, updatedProject.Description, description)
	th.AssertEquals(t, updatedProject.Enabled, iFalse)
}

func TestProjectsDomain(t *testing.T) {
	clients.RequireAdmin(t)

	client, err := clients.NewIdentityV3Client()
	th.AssertNoErr(t, err)

	var iTrue = true
	createOpts := projects.CreateOpts{
		IsDomain: &iTrue,
	}

	projectDomain, err := CreateProject(t, client, &createOpts)
	th.AssertNoErr(t, err)
	defer DeleteProject(t, client, projectDomain.ID)

	tools.PrintResource(t, projectDomain)

	createOpts = projects.CreateOpts{
		DomainID: projectDomain.ID,
	}

	project, err := CreateProject(t, client, &createOpts)
	th.AssertNoErr(t, err)
	defer DeleteProject(t, client, project.ID)

	tools.PrintResource(t, project)

	th.AssertEquals(t, project.DomainID, projectDomain.ID)

	var iFalse = false
	updateOpts := projects.UpdateOpts{
		Enabled: &iFalse,
	}

	_, err = projects.Update(client, projectDomain.ID, updateOpts).Extract()
	th.AssertNoErr(t, err)
}

func TestProjectsNested(t *testing.T) {
	clients.RequireAdmin(t)

	client, err := clients.NewIdentityV3Client()
	th.AssertNoErr(t, err)

	projectMain, err := CreateProject(t, client, nil)
	th.AssertNoErr(t, err)
	defer DeleteProject(t, client, projectMain.ID)

	tools.PrintResource(t, projectMain)

	createOpts := projects.CreateOpts{
		ParentID: projectMain.ID,
	}

	project, err := CreateProject(t, client, &createOpts)
	th.AssertNoErr(t, err)
	defer DeleteProject(t, client, project.ID)

	tools.PrintResource(t, project)

	th.AssertEquals(t, project.ParentID, projectMain.ID)
}
