package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/identity/v2/users"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	MockListUserResponse(t)

	count := 0

	err := users.List(client.ServiceClient()).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := users.ExtractUsers(page)
		th.AssertNoErr(t, err)

		expected := []users.User{
			{
				ID:       "u1000",
				Name:     "John Smith",
				Username: "jqsmith",
				Email:    "john.smith@example.org",
				Enabled:  true,
				TenantID: "12345",
			},
			{
				ID:       "u1001",
				Name:     "Jane Smith",
				Username: "jqsmith",
				Email:    "jane.smith@example.org",
				Enabled:  true,
				TenantID: "12345",
			},
		}
		th.CheckDeepEquals(t, expected, actual)
		return true, nil
	})
	th.AssertNoErr(t, err)
	th.AssertEquals(t, 1, count)
}

func TestCreateUser(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockCreateUserResponse(t)

	opts := users.CreateOpts{
		Name:     "new_user",
		TenantID: "12345",
		Enabled:  gophercloud.Disabled,
		Email:    "new_user@foo.com",
	}

	user, err := users.Create(client.ServiceClient(), opts).Extract()

	th.AssertNoErr(t, err)

	expected := &users.User{
		Name:     "new_user",
		ID:       "c39e3de9be2d4c779f1dfd6abacc176d",
		Email:    "new_user@foo.com",
		Enabled:  false,
		TenantID: "12345",
	}

	th.AssertDeepEquals(t, expected, user)
}

func TestGetUser(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockGetUserResponse(t)

	user, err := users.Get(client.ServiceClient(), "new_user").Extract()
	th.AssertNoErr(t, err)

	expected := &users.User{
		Name:     "new_user",
		ID:       "c39e3de9be2d4c779f1dfd6abacc176d",
		Email:    "new_user@foo.com",
		Enabled:  false,
		TenantID: "12345",
	}

	th.AssertDeepEquals(t, expected, user)
}

func TestUpdateUser(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockUpdateUserResponse(t)

	id := "c39e3de9be2d4c779f1dfd6abacc176d"
	opts := users.UpdateOpts{
		Name:    "new_name",
		Enabled: gophercloud.Enabled,
		Email:   "new_email@foo.com",
	}

	user, err := users.Update(client.ServiceClient(), id, opts).Extract()

	th.AssertNoErr(t, err)

	expected := &users.User{
		Name:     "new_name",
		ID:       id,
		Email:    "new_email@foo.com",
		Enabled:  true,
		TenantID: "12345",
	}

	th.AssertDeepEquals(t, expected, user)
}

func TestDeleteUser(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockDeleteUserResponse(t)

	res := users.Delete(client.ServiceClient(), "c39e3de9be2d4c779f1dfd6abacc176d")
	th.AssertNoErr(t, res.Err)
}

func TestListingUserRoles(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockListRolesResponse(t)

	tenantID := "1d8b6120dcc640fda4fc9194ffc80273"
	userID := "c39e3de9be2d4c779f1dfd6abacc176d"

	err := users.ListRoles(client.ServiceClient(), tenantID, userID).EachPage(func(page pagination.Page) (bool, error) {
		actual, err := users.ExtractRoles(page)
		th.AssertNoErr(t, err)

		expected := []users.Role{
			{ID: "9fe2ff9ee4384b1894a90878d3e92bab", Name: "foo_role"},
			{ID: "1ea3d56793574b668e85960fbf651e13", Name: "admin"},
		}

		th.CheckDeepEquals(t, expected, actual)

		return true, nil
	})

	th.AssertNoErr(t, err)
}
