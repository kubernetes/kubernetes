// Package v2 contains common functions for creating identity-based resources
// for use in acceptance tests. See the `*_test.go` files for example usages.
package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/identity/v2/extensions/admin/roles"
	"github.com/gophercloud/gophercloud/openstack/identity/v2/tenants"
	"github.com/gophercloud/gophercloud/openstack/identity/v2/tokens"
	"github.com/gophercloud/gophercloud/openstack/identity/v2/users"
)

// AddUserRole will grant a role to a user in a tenant. An error will be
// returned if the grant was unsuccessful.
func AddUserRole(t *testing.T, client *gophercloud.ServiceClient, tenant *tenants.Tenant, user *users.User, role *roles.Role) error {
	t.Logf("Attempting to grant user %s role %s in tenant %s", user.ID, role.ID, tenant.ID)

	err := roles.AddUser(client, tenant.ID, user.ID, role.ID).ExtractErr()
	if err != nil {
		return err
	}

	t.Logf("Granted user %s role %s in tenant %s", user.ID, role.ID, tenant.ID)

	return nil
}

// CreateUser will create a user with a random name and adds them to the given
// tenant. An error will be returned if the user was unable to be created.
func CreateUser(t *testing.T, client *gophercloud.ServiceClient, tenant *tenants.Tenant) (*users.User, error) {
	userName := tools.RandomString("user_", 5)
	userEmail := userName + "@foo.com"
	t.Logf("Creating user: %s", userName)

	createOpts := users.CreateOpts{
		Name:     userName,
		Enabled:  gophercloud.Disabled,
		TenantID: tenant.ID,
		Email:    userEmail,
	}

	user, err := users.Create(client, createOpts).Extract()
	if err != nil {
		return user, err
	}

	return user, nil
}

// DeleteUser will delete a user. A fatal error will occur if the delete was
// unsuccessful. This works best when used as a deferred function.
func DeleteUser(t *testing.T, client *gophercloud.ServiceClient, user *users.User) {
	t.Logf("Attempting to delete user: %s", user.Name)

	result := users.Delete(client, user.ID)
	if result.Err != nil {
		t.Fatalf("Unable to delete user")
	}

	t.Logf("Deleted user: %s", user.Name)
}

// DeleteUserRole will revoke a role of a user in a tenant. A fatal error will
// occur if the revoke was unsuccessful. This works best when used as a
// deferred function.
func DeleteUserRole(t *testing.T, client *gophercloud.ServiceClient, tenant *tenants.Tenant, user *users.User, role *roles.Role) {
	t.Logf("Attempting to remove role %s from user %s in tenant %s", role.ID, user.ID, tenant.ID)

	err := roles.DeleteUser(client, tenant.ID, user.ID, role.ID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to remove role")
	}

	t.Logf("Removed role %s from user %s in tenant %s", role.ID, user.ID, tenant.ID)
}

// FindRole finds all roles that the current authenticated client has access
// to and returns the first one found. An error will be returned if the lookup
// was unsuccessful.
func FindRole(t *testing.T, client *gophercloud.ServiceClient) (*roles.Role, error) {
	var role *roles.Role

	allPages, err := roles.List(client).AllPages()
	if err != nil {
		return role, err
	}

	allRoles, err := roles.ExtractRoles(allPages)
	if err != nil {
		return role, err
	}

	for _, r := range allRoles {
		role = &r
		break
	}

	return role, nil
}

// FindTenant finds all tenants that the current authenticated client has access
// to and returns the first one found. An error will be returned if the lookup
// was unsuccessful.
func FindTenant(t *testing.T, client *gophercloud.ServiceClient) (*tenants.Tenant, error) {
	var tenant *tenants.Tenant

	allPages, err := tenants.List(client, nil).AllPages()
	if err != nil {
		return tenant, err
	}

	allTenants, err := tenants.ExtractTenants(allPages)
	if err != nil {
		return tenant, err
	}

	for _, t := range allTenants {
		tenant = &t
		break
	}

	return tenant, nil
}

// UpdateUser will update an existing user with a new randomly generated name.
// An error will be returned if the update was unsuccessful.
func UpdateUser(t *testing.T, client *gophercloud.ServiceClient, user *users.User) (*users.User, error) {
	userName := tools.RandomString("user_", 5)
	userEmail := userName + "@foo.com"

	t.Logf("Attempting to update user name from %s to %s", user.Name, userName)

	updateOpts := users.UpdateOpts{
		Name:  userName,
		Email: userEmail,
	}

	newUser, err := users.Update(client, user.ID, updateOpts).Extract()
	if err != nil {
		return newUser, err
	}

	return newUser, nil
}

// PrintCatalogEntry will print a catalog entry and all of its attributes.
func PrintCatalogEntry(t *testing.T, catalogEntry *tokens.CatalogEntry) {
	t.Logf("Name: %s", catalogEntry.Name)
	t.Logf("Type: %s", catalogEntry.Type)

	t.Log("Endpoints:")
	for _, endpoint := range catalogEntry.Endpoints {
		t.Logf("\tTenantID: %s", endpoint.TenantID)
		t.Logf("\tPublicURL: %s", endpoint.PublicURL)
		t.Logf("\tInternalURL: %s", endpoint.InternalURL)
		t.Logf("\tAdminURL: %s", endpoint.AdminURL)
		t.Logf("\tRegion: %s", endpoint.Region)
		t.Logf("\tVersionID: %s", endpoint.VersionID)
		t.Logf("\tVersionInfo: %s", endpoint.VersionInfo)
		t.Logf("\tVersionList: %s", endpoint.VersionList)
	}
}

// PrintRole will print a role and all of its attributes.
func PrintRole(t *testing.T, role *roles.Role) {
	t.Logf("ID: %s", role.ID)
	t.Logf("Name: %v", role.Name)
	t.Logf("Description: %s", role.Description)
	t.Logf("ServiceID: %s", role.ServiceID)
}

// PrintTenant will print a tenant and all of its attributes.
func PrintTenant(t *testing.T, tenant *tenants.Tenant) {
	t.Logf("ID: %s", tenant.ID)
	t.Logf("Name: %s", tenant.Name)
	t.Logf("Description: %s", tenant.Description)
	t.Logf("Enabled: %t", tenant.Enabled)
}

// PrintToken will print a token and all of its attributes.
func PrintToken(t *testing.T, token *tokens.Token) {
	t.Logf("ID: %s", token.ID)
	t.Logf("ExpiresAt: %v", token.ExpiresAt)
	t.Logf("TenantID: %s", token.Tenant.ID)
}

// PrintTokenUser will print the user information of a token and all attributes.
func PrintTokenUser(t *testing.T, user *tokens.User) {
	t.Logf("ID: %s", user.ID)
	t.Logf("Name: %s", user.Name)
	t.Logf("Username: %s", user.UserName)

	t.Log("Roles")
	for _, role := range user.Roles {
		t.Logf("\t%s", role)
	}
}

// PrintUser will print a user and all of its attributes.
func PrintUser(t *testing.T, user *users.User) {
	t.Logf("ID: %s", user.ID)
	t.Logf("Name: %s", user.Name)
	t.Logf("Username: %s", user.Username)
	t.Logf("Enabled: %t", user.Enabled)
	t.Logf("Email: %s", user.Email)
	t.Logf("TenantID: %s", user.TenantID)
}

// PrintUserRole will print the roles that a user has been granted.
func PrintUserRole(t *testing.T, role *users.Role) {
	t.Logf("ID: %s", role.ID)
	t.Logf("Name: %s", role.Name)
}
