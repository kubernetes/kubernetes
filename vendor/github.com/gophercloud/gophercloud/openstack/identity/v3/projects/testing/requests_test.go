package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud/openstack/identity/v3/projects"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

func TestListProjects(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListProjectsSuccessfully(t)

	count := 0
	err := projects.List(client.ServiceClient(), nil).EachPage(func(page pagination.Page) (bool, error) {
		count++

		actual, err := projects.ExtractProjects(page)
		th.AssertNoErr(t, err)

		th.CheckDeepEquals(t, ExpectedProjectSlice, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.CheckEquals(t, count, 1)
}

func TestGetProject(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleGetProjectSuccessfully(t)

	actual, err := projects.Get(client.ServiceClient(), "1234").Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, RedTeam, *actual)
}

func TestCreateProject(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleCreateProjectSuccessfully(t)

	createOpts := projects.CreateOpts{
		Name:        "Red Team",
		Description: "The team that is red",
	}

	actual, err := projects.Create(client.ServiceClient(), createOpts).Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, RedTeam, *actual)
}

func TestDeleteProject(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleDeleteProjectSuccessfully(t)

	res := projects.Delete(client.ServiceClient(), "1234")
	th.AssertNoErr(t, res.Err)
}

func TestUpdateProject(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleUpdateProjectSuccessfully(t)

	updateOpts := projects.UpdateOpts{
		Name:        "Bright Red Team",
		Description: "The team that is bright red",
	}

	actual, err := projects.Update(client.ServiceClient(), "1234", updateOpts).Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, UpdatedRedTeam, *actual)
}
