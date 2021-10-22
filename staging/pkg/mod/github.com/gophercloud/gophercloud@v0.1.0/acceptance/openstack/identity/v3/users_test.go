// +build acceptance

package v3

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/identity/v3/groups"
	"github.com/gophercloud/gophercloud/openstack/identity/v3/projects"
	"github.com/gophercloud/gophercloud/openstack/identity/v3/users"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestUsersList(t *testing.T) {
	clients.RequireAdmin(t)

	client, err := clients.NewIdentityV3Client()
	th.AssertNoErr(t, err)

	var iTrue bool = true
	listOpts := users.ListOpts{
		Enabled: &iTrue,
	}

	allPages, err := users.List(client, listOpts).AllPages()
	th.AssertNoErr(t, err)

	allUsers, err := users.ExtractUsers(allPages)
	th.AssertNoErr(t, err)

	var found bool
	for _, user := range allUsers {
		tools.PrintResource(t, user)
		tools.PrintResource(t, user.Extra)

		if user.Name == "admin" {
			found = true
		}
	}

	th.AssertEquals(t, found, true)

	listOpts.Filters = map[string]string{
		"name__contains": "dmi",
	}

	allPages, err = users.List(client, listOpts).AllPages()
	th.AssertNoErr(t, err)

	allUsers, err = users.ExtractUsers(allPages)
	th.AssertNoErr(t, err)

	found = false
	for _, user := range allUsers {
		tools.PrintResource(t, user)
		tools.PrintResource(t, user.Extra)

		if user.Name == "admin" {
			found = true
		}
	}

	th.AssertEquals(t, found, true)

	listOpts.Filters = map[string]string{
		"name__contains": "foo",
	}

	allPages, err = users.List(client, listOpts).AllPages()
	th.AssertNoErr(t, err)

	allUsers, err = users.ExtractUsers(allPages)
	th.AssertNoErr(t, err)

	found = false
	for _, user := range allUsers {
		tools.PrintResource(t, user)
		tools.PrintResource(t, user.Extra)

		if user.Name == "admin" {
			found = true
		}
	}

	th.AssertEquals(t, found, false)
}

func TestUsersGet(t *testing.T) {
	clients.RequireAdmin(t)

	client, err := clients.NewIdentityV3Client()
	th.AssertNoErr(t, err)

	allPages, err := users.List(client, nil).AllPages()
	th.AssertNoErr(t, err)

	allUsers, err := users.ExtractUsers(allPages)
	th.AssertNoErr(t, err)

	user := allUsers[0]
	p, err := users.Get(client, user.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, p)

	th.AssertEquals(t, user.Name, p.Name)
}

func TestUserCRUD(t *testing.T) {
	clients.RequireAdmin(t)

	client, err := clients.NewIdentityV3Client()
	th.AssertNoErr(t, err)

	project, err := CreateProject(t, client, nil)
	th.AssertNoErr(t, err)
	defer DeleteProject(t, client, project.ID)

	tools.PrintResource(t, project)

	createOpts := users.CreateOpts{
		DefaultProjectID: project.ID,
		Description:      "test description",
		Password:         "foobar",
		DomainID:         "default",
		Options: map[users.Option]interface{}{
			users.IgnorePasswordExpiry: true,
			users.MultiFactorAuthRules: []interface{}{
				[]string{"password", "totp"},
				[]string{"password", "custom-auth-method"},
			},
		},
		Extra: map[string]interface{}{
			"email": "jsmith@example.com",
		},
	}

	user, err := CreateUser(t, client, &createOpts)
	th.AssertNoErr(t, err)
	defer DeleteUser(t, client, user.ID)

	tools.PrintResource(t, user)
	tools.PrintResource(t, user.Extra)

	th.AssertEquals(t, user.Description, createOpts.Description)
	th.AssertEquals(t, user.DomainID, createOpts.DomainID)

	iFalse := false
	name := "newtestuser"
	description := ""
	updateOpts := users.UpdateOpts{
		Name:        name,
		Description: &description,
		Enabled:     &iFalse,
		Options: map[users.Option]interface{}{
			users.MultiFactorAuthRules: nil,
		},
		Extra: map[string]interface{}{
			"disabled_reason": "DDOS",
		},
	}

	newUser, err := users.Update(client, user.ID, updateOpts).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newUser)
	tools.PrintResource(t, newUser.Extra)

	th.AssertEquals(t, newUser.Name, name)
	th.AssertEquals(t, newUser.Description, description)
	th.AssertEquals(t, newUser.Enabled, iFalse)
	th.AssertEquals(t, newUser.Extra["disabled_reason"], "DDOS")
}

func TestUserChangePassword(t *testing.T) {
	clients.RequireAdmin(t)

	client, err := clients.NewIdentityV3Client()
	th.AssertNoErr(t, err)

	createOpts := users.CreateOpts{
		Password: "secretsecret",
		DomainID: "default",
	}

	user, err := CreateUser(t, client, &createOpts)
	th.AssertNoErr(t, err)
	defer DeleteUser(t, client, user.ID)

	tools.PrintResource(t, user)
	tools.PrintResource(t, user.Extra)

	changePasswordOpts := users.ChangePasswordOpts{
		OriginalPassword: "secretsecret",
		Password:         "new_secretsecret",
	}
	err = users.ChangePassword(client, user.ID, changePasswordOpts).ExtractErr()
	th.AssertNoErr(t, err)
}

func TestUsersGroups(t *testing.T) {
	clients.RequireAdmin(t)

	client, err := clients.NewIdentityV3Client()
	th.AssertNoErr(t, err)

	createOpts := users.CreateOpts{
		Password: "foobar",
		DomainID: "default",
	}

	user, err := CreateUser(t, client, &createOpts)
	th.AssertNoErr(t, err)
	defer DeleteUser(t, client, user.ID)

	tools.PrintResource(t, user)
	tools.PrintResource(t, user.Extra)

	createGroupOpts := groups.CreateOpts{
		Name:     "testgroup",
		DomainID: "default",
	}

	// Create Group in the default domain
	group, err := CreateGroup(t, client, &createGroupOpts)
	th.AssertNoErr(t, err)
	defer DeleteGroup(t, client, group.ID)

	tools.PrintResource(t, group)
	tools.PrintResource(t, group.Extra)

	err = users.AddToGroup(client, group.ID, user.ID).ExtractErr()
	th.AssertNoErr(t, err)

	allGroupPages, err := users.ListGroups(client, user.ID).AllPages()
	th.AssertNoErr(t, err)

	allGroups, err := groups.ExtractGroups(allGroupPages)
	th.AssertNoErr(t, err)

	var found bool
	for _, g := range allGroups {
		tools.PrintResource(t, g)
		tools.PrintResource(t, g.Extra)

		if g.ID == group.ID {
			found = true
		}
	}

	th.AssertEquals(t, found, true)

	found = false
	allUserPages, err := users.ListInGroup(client, group.ID, nil).AllPages()
	th.AssertNoErr(t, err)

	allUsers, err := users.ExtractUsers(allUserPages)
	th.AssertNoErr(t, err)

	for _, u := range allUsers {
		tools.PrintResource(t, user)
		tools.PrintResource(t, user.Extra)

		if u.ID == user.ID {
			found = true
		}
	}

	th.AssertEquals(t, found, true)

	ok, err := users.IsMemberOfGroup(client, group.ID, user.ID).Extract()
	if err != nil {
		t.Fatalf("Unable to check whether user belongs to group: %v", err)
	}
	if !ok {
		t.Fatalf("User %s is expected to be a member of group %s", user.ID, group.ID)
	}

	err = users.RemoveFromGroup(client, group.ID, user.ID).ExtractErr()
	th.AssertNoErr(t, err)

	allGroupPages, err = users.ListGroups(client, user.ID).AllPages()
	th.AssertNoErr(t, err)

	allGroups, err = groups.ExtractGroups(allGroupPages)
	th.AssertNoErr(t, err)

	found = false
	for _, g := range allGroups {
		tools.PrintResource(t, g)
		tools.PrintResource(t, g.Extra)

		if g.ID == group.ID {
			found = true
		}
	}

	th.AssertEquals(t, found, false)

	found = false
	allUserPages, err = users.ListInGroup(client, group.ID, nil).AllPages()
	th.AssertNoErr(t, err)

	allUsers, err = users.ExtractUsers(allUserPages)
	th.AssertNoErr(t, err)

	for _, u := range allUsers {
		tools.PrintResource(t, user)
		tools.PrintResource(t, user.Extra)

		if u.ID == user.ID {
			found = true
		}
	}

	th.AssertEquals(t, found, false)

}

func TestUsersListProjects(t *testing.T) {
	clients.RequireAdmin(t)

	client, err := clients.NewIdentityV3Client()
	th.AssertNoErr(t, err)

	allUserPages, err := users.List(client, nil).AllPages()
	th.AssertNoErr(t, err)

	allUsers, err := users.ExtractUsers(allUserPages)
	th.AssertNoErr(t, err)

	user := allUsers[0]

	allProjectPages, err := users.ListProjects(client, user.ID).AllPages()
	th.AssertNoErr(t, err)

	allProjects, err := projects.ExtractProjects(allProjectPages)
	th.AssertNoErr(t, err)

	for _, project := range allProjects {
		tools.PrintResource(t, project)
	}
}
