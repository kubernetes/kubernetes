package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud/openstack/identity/v3/groups"
	"github.com/gophercloud/gophercloud/openstack/identity/v3/projects"
	"github.com/gophercloud/gophercloud/openstack/identity/v3/users"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

func TestListUsers(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListUsersSuccessfully(t)

	count := 0
	err := users.List(client.ServiceClient(), nil).EachPage(func(page pagination.Page) (bool, error) {
		count++

		actual, err := users.ExtractUsers(page)
		th.AssertNoErr(t, err)

		th.CheckDeepEquals(t, ExpectedUsersSlice, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.CheckEquals(t, count, 1)
}

func TestListUsersAllPages(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListUsersSuccessfully(t)

	allPages, err := users.List(client.ServiceClient(), nil).AllPages()
	th.AssertNoErr(t, err)
	actual, err := users.ExtractUsers(allPages)
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, ExpectedUsersSlice, actual)
	th.AssertEquals(t, ExpectedUsersSlice[0].Extra["email"], "glance@localhost")
	th.AssertEquals(t, ExpectedUsersSlice[1].Extra["email"], "jsmith@example.com")
}

func TestListUsersFiltersCheck(t *testing.T) {
	type test struct {
		filterName string
		wantErr    bool
	}
	tests := []test{
		{"foo__contains", false},
		{"foo", true},
		{"foo_contains", true},
		{"foo__", true},
		{"__foo", true},
	}

	var listOpts users.ListOpts
	for _, _test := range tests {
		listOpts.Filters = map[string]string{_test.filterName: "bar"}
		_, err := listOpts.ToUserListQuery()

		if !_test.wantErr {
			th.AssertNoErr(t, err)
		} else {
			switch _t := err.(type) {
			case nil:
				t.Fatal("error expected but got a nil")
			case users.InvalidListFilter:
			default:
				t.Fatalf("unexpected error type: [%T]", _t)
			}
		}
	}
}

func TestGetUser(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleGetUserSuccessfully(t)

	actual, err := users.Get(client.ServiceClient(), "9fe1d3").Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, SecondUser, *actual)
	th.AssertEquals(t, SecondUser.Extra["email"], "jsmith@example.com")
}

func TestCreateUser(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleCreateUserSuccessfully(t)

	iTrue := true
	createOpts := users.CreateOpts{
		Name:             "jsmith",
		DomainID:         "1789d1",
		Enabled:          &iTrue,
		Password:         "secretsecret",
		DefaultProjectID: "263fd9",
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

	actual, err := users.Create(client.ServiceClient(), createOpts).Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, SecondUser, *actual)
}

func TestCreateNoOptionsUser(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleCreateNoOptionsUserSuccessfully(t)

	iTrue := true
	createOpts := users.CreateOpts{
		Name:             "jsmith",
		DomainID:         "1789d1",
		Enabled:          &iTrue,
		Password:         "secretsecret",
		DefaultProjectID: "263fd9",
		Extra: map[string]interface{}{
			"email": "jsmith@example.com",
		},
	}

	actual, err := users.Create(client.ServiceClient(), createOpts).Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, SecondUserNoOptions, *actual)
}

func TestUpdateUser(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleUpdateUserSuccessfully(t)

	iFalse := false
	updateOpts := users.UpdateOpts{
		Enabled: &iFalse,
		Options: map[users.Option]interface{}{
			users.MultiFactorAuthRules: nil,
		},
		Extra: map[string]interface{}{
			"disabled_reason": "DDOS",
		},
	}

	actual, err := users.Update(client.ServiceClient(), "9fe1d3", updateOpts).Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, SecondUserUpdated, *actual)
}

func TestChangeUserPassword(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleChangeUserPasswordSuccessfully(t)

	changePasswordOpts := users.ChangePasswordOpts{
		OriginalPassword: "secretsecret",
		Password:         "new_secretsecret",
	}

	res := users.ChangePassword(client.ServiceClient(), "9fe1d3", changePasswordOpts)
	th.AssertNoErr(t, res.Err)
}

func TestDeleteUser(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleDeleteUserSuccessfully(t)

	res := users.Delete(client.ServiceClient(), "9fe1d3")
	th.AssertNoErr(t, res.Err)
}

func TestListUserGroups(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListUserGroupsSuccessfully(t)
	allPages, err := users.ListGroups(client.ServiceClient(), "9fe1d3").AllPages()
	th.AssertNoErr(t, err)
	actual, err := groups.ExtractGroups(allPages)
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, ExpectedGroupsSlice, actual)
}

func TestAddToGroup(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleAddToGroupSuccessfully(t)
	res := users.AddToGroup(client.ServiceClient(), "ea167b", "9fe1d3")
	th.AssertNoErr(t, res.Err)
}

func TestIsMemberOfGroup(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleIsMemberOfGroupSuccessfully(t)
	ok, err := users.IsMemberOfGroup(client.ServiceClient(), "ea167b", "9fe1d3").Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, true, ok)
}

func TestRemoveFromGroup(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleRemoveFromGroupSuccessfully(t)
	res := users.RemoveFromGroup(client.ServiceClient(), "ea167b", "9fe1d3")
	th.AssertNoErr(t, res.Err)
}

func TestListUserProjects(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListUserProjectsSuccessfully(t)
	allPages, err := users.ListProjects(client.ServiceClient(), "9fe1d3").AllPages()
	th.AssertNoErr(t, err)
	actual, err := projects.ExtractProjects(allPages)
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, ExpectedProjectsSlice, actual)
}

func TestListInGroup(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListInGroupSuccessfully(t)

	iTrue := true
	listOpts := users.ListOpts{
		Enabled: &iTrue,
	}

	allPages, err := users.ListInGroup(client.ServiceClient(), "ea167b", listOpts).AllPages()
	th.AssertNoErr(t, err)
	actual, err := users.ExtractUsers(allPages)
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, ExpectedUsersSlice, actual)
}
