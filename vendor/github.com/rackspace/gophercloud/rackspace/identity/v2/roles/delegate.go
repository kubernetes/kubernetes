package roles

import (
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"

	os "github.com/rackspace/gophercloud/openstack/identity/v2/extensions/admin/roles"
)

// List is the operation responsible for listing all available global roles
// that a user can adopt.
func List(client *gophercloud.ServiceClient) pagination.Pager {
	return os.List(client)
}

// AddUserRole is the operation responsible for assigning a particular role to
// a user. This is confined to the scope of the user's tenant - so the tenant
// ID is a required argument.
func AddUserRole(client *gophercloud.ServiceClient, userID, roleID string) UserRoleResult {
	var result UserRoleResult

	_, result.Err = client.Request("PUT", userRoleURL(client, userID, roleID), gophercloud.RequestOpts{
		OkCodes: []int{200, 201},
	})

	return result
}

// DeleteUserRole is the operation responsible for deleting a particular role
// from a user. This is confined to the scope of the user's tenant - so the
// tenant ID is a required argument.
func DeleteUserRole(client *gophercloud.ServiceClient, userID, roleID string) UserRoleResult {
	var result UserRoleResult

	_, result.Err = client.Request("DELETE", userRoleURL(client, userID, roleID), gophercloud.RequestOpts{
		OkCodes: []int{204},
	})

	return result
}

// UserRoleResult represents the result of either an AddUserRole or
// a DeleteUserRole operation.
type UserRoleResult struct {
	gophercloud.ErrResult
}

func userRoleURL(c *gophercloud.ServiceClient, userID, roleID string) string {
	return c.ServiceURL(os.UserPath, userID, os.RolePath, os.ExtPath, roleID)
}
