// +build acceptance identity roles

package v2

import (
	"testing"

	"github.com/rackspace/gophercloud"
	os "github.com/rackspace/gophercloud/openstack/identity/v2/extensions/admin/roles"

	"github.com/rackspace/gophercloud/pagination"
	"github.com/rackspace/gophercloud/rackspace/identity/v2/roles"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestRoles(t *testing.T) {
	client := authenticatedClient(t)

	userID := createUser(t, client)
	roleID := listRoles(t, client)

	addUserRole(t, client, userID, roleID)

	deleteUserRole(t, client, userID, roleID)

	deleteUser(t, client, userID)
}

func listRoles(t *testing.T, client *gophercloud.ServiceClient) string {
	var roleID string

	err := roles.List(client).EachPage(func(page pagination.Page) (bool, error) {
		roleList, err := os.ExtractRoles(page)
		th.AssertNoErr(t, err)

		for _, role := range roleList {
			t.Logf("Listing role: ID [%s] Name [%s]", role.ID, role.Name)
			roleID = role.ID
		}

		return true, nil
	})

	th.AssertNoErr(t, err)

	return roleID
}

func addUserRole(t *testing.T, client *gophercloud.ServiceClient, userID, roleID string) {
	err := roles.AddUserRole(client, userID, roleID).ExtractErr()
	th.AssertNoErr(t, err)
	t.Logf("Added role %s to user %s", roleID, userID)
}

func deleteUserRole(t *testing.T, client *gophercloud.ServiceClient, userID, roleID string) {
	err := roles.DeleteUserRole(client, userID, roleID).ExtractErr()
	th.AssertNoErr(t, err)
	t.Logf("Removed role %s from user %s", roleID, userID)
}
