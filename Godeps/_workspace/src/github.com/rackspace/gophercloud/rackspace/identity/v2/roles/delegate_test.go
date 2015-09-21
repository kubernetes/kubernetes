package roles

import (
	"testing"

	os "github.com/rackspace/gophercloud/openstack/identity/v2/extensions/admin/roles"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
	"github.com/rackspace/gophercloud/testhelper/client"
)

func TestRole(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	MockListRoleResponse(t)

	count := 0

	err := List(client.ServiceClient()).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := os.ExtractRoles(page)
		if err != nil {
			t.Errorf("Failed to extract users: %v", err)
			return false, err
		}

		expected := []os.Role{
			os.Role{
				ID:          "123",
				Name:        "compute:admin",
				Description: "Nova Administrator",
				ServiceID:   "cke5372ebabeeabb70a0e702a4626977x4406e5",
			},
		}

		th.CheckDeepEquals(t, expected, actual)

		return true, nil
	})

	th.AssertNoErr(t, err)
	th.AssertEquals(t, 1, count)
}

func TestAddUserRole(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	MockAddUserRoleResponse(t)

	err := AddUserRole(client.ServiceClient(), "{user_id}", "{role_id}").ExtractErr()

	th.AssertNoErr(t, err)
}

func TestDeleteUserRole(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	MockDeleteUserRoleResponse(t)

	err := DeleteUserRole(client.ServiceClient(), "{user_id}", "{role_id}").ExtractErr()

	th.AssertNoErr(t, err)
}
