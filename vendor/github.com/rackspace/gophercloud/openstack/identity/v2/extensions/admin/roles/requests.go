package roles

import (
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// List is the operation responsible for listing all available global roles
// that a user can adopt.
func List(client *gophercloud.ServiceClient) pagination.Pager {
	createPage := func(r pagination.PageResult) pagination.Page {
		return RolePage{pagination.SinglePageBase(r)}
	}
	return pagination.NewPager(client, rootURL(client), createPage)
}

// AddUserRole is the operation responsible for assigning a particular role to
// a user. This is confined to the scope of the user's tenant - so the tenant
// ID is a required argument.
func AddUserRole(client *gophercloud.ServiceClient, tenantID, userID, roleID string) UserRoleResult {
	var result UserRoleResult
	_, result.Err = client.Put(userRoleURL(client, tenantID, userID, roleID), nil, nil, nil)
	return result
}

// DeleteUserRole is the operation responsible for deleting a particular role
// from a user. This is confined to the scope of the user's tenant - so the
// tenant ID is a required argument.
func DeleteUserRole(client *gophercloud.ServiceClient, tenantID, userID, roleID string) UserRoleResult {
	var result UserRoleResult
	_, result.Err = client.Delete(userRoleURL(client, tenantID, userID, roleID), nil)
	return result
}
