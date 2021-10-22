// Package v2 contains common functions for creating identity-based resources
// for use in acceptance tests. See the `*_test.go` files for example usages.
package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/identity/v2/extensions/admin/roles"
	"github.com/gophercloud/gophercloud/openstack/identity/v2/tenants"
	"github.com/gophercloud/gophercloud/openstack/identity/v2/users"
	th "github.com/gophercloud/gophercloud/testhelper"
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

// CreateTenant will create a project with a random name.
// It takes an optional createOpts parameter since creating a project
// has so many options. An error will be returned if the project was
// unable to be created.
func CreateTenant(t *testing.T, client *gophercloud.ServiceClient, c *tenants.CreateOpts) (*tenants.Tenant, error) {
	name := tools.RandomString("ACPTTEST", 8)
	description := tools.RandomString("ACPTTEST-DESC", 8)
	t.Logf("Attempting to create tenant: %s", name)

	var createOpts tenants.CreateOpts
	if c != nil {
		createOpts = *c
	} else {
		createOpts = tenants.CreateOpts{}
	}

	createOpts.Name = name
	createOpts.Description = description

	tenant, err := tenants.Create(client, createOpts).Extract()
	if err != nil {
		return tenant, err
	}

	t.Logf("Successfully created project %s with ID %s", name, tenant.ID)

	th.AssertEquals(t, name, tenant.Name)
	th.AssertEquals(t, description, tenant.Description)

	return tenant, nil
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

	th.AssertEquals(t, userName, user.Name)

	return user, nil
}

// DeleteTenant will delete a tenant by ID. A fatal error will occur if
// the tenant ID failed to be deleted. This works best when using it as
// a deferred function.
func DeleteTenant(t *testing.T, client *gophercloud.ServiceClient, tenantID string) {
	err := tenants.Delete(client, tenantID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete tenant %s: %v", tenantID, err)
	}

	t.Logf("Deleted tenant: %s", tenantID)
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

	th.AssertEquals(t, userName, newUser.Name)

	return newUser, nil
}
