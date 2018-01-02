// +build acceptance identity roles

package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/identity/v2/extensions/admin/roles"
	"github.com/gophercloud/gophercloud/openstack/identity/v2/users"
)

func TestRolesAddToUser(t *testing.T) {
	client, err := clients.NewIdentityV2AdminClient()
	if err != nil {
		t.Fatalf("Unable to obtain an identity client: %v", err)
	}

	tenant, err := FindTenant(t, client)
	if err != nil {
		t.Fatalf("Unable to get a tenant: %v", err)
	}

	role, err := FindRole(t, client)
	if err != nil {
		t.Fatalf("Unable to get a role: %v", err)
	}

	user, err := CreateUser(t, client, tenant)
	if err != nil {
		t.Fatalf("Unable to create a user: %v", err)
	}
	defer DeleteUser(t, client, user)

	err = AddUserRole(t, client, tenant, user, role)
	if err != nil {
		t.Fatalf("Unable to add role to user: %v", err)
	}
	defer DeleteUserRole(t, client, tenant, user, role)

	allPages, err := users.ListRoles(client, tenant.ID, user.ID).AllPages()
	if err != nil {
		t.Fatalf("Unable to obtain roles for user: %v", err)
	}

	allRoles, err := users.ExtractRoles(allPages)
	if err != nil {
		t.Fatalf("Unable to extract roles: %v", err)
	}

	t.Logf("Roles of user %s:", user.Name)
	for _, role := range allRoles {
		tools.PrintResource(t, role)
	}
}

func TestRolesList(t *testing.T) {
	client, err := clients.NewIdentityV2AdminClient()
	if err != nil {
		t.Fatalf("Unable to create an identity client: %v", err)
	}

	allPages, err := roles.List(client).AllPages()
	if err != nil {
		t.Fatalf("Unable to list all roles: %v", err)
	}

	allRoles, err := roles.ExtractRoles(allPages)
	if err != nil {
		t.Fatalf("Unable to extract roles: %v", err)
	}

	for _, r := range allRoles {
		tools.PrintResource(t, r)
	}
}
