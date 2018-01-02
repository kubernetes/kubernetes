// +build acceptance identity

package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/identity/v2/users"
)

func TestUsersList(t *testing.T) {
	client, err := clients.NewIdentityV2AdminClient()
	if err != nil {
		t.Fatalf("Unable to obtain an identity client: %v", err)
	}

	allPages, err := users.List(client).AllPages()
	if err != nil {
		t.Fatalf("Unable to list users: %v", err)
	}

	allUsers, err := users.ExtractUsers(allPages)
	if err != nil {
		t.Fatalf("Unable to extract users: %v", err)
	}

	for _, user := range allUsers {
		tools.PrintResource(t, user)
	}
}

func TestUsersCreateUpdateDelete(t *testing.T) {
	client, err := clients.NewIdentityV2AdminClient()
	if err != nil {
		t.Fatalf("Unable to obtain an identity client: %v", err)
	}

	tenant, err := FindTenant(t, client)
	if err != nil {
		t.Fatalf("Unable to get a tenant: %v", err)
	}

	user, err := CreateUser(t, client, tenant)
	if err != nil {
		t.Fatalf("Unable to create a user: %v", err)
	}
	defer DeleteUser(t, client, user)

	tools.PrintResource(t, user)

	newUser, err := UpdateUser(t, client, user)
	if err != nil {
		t.Fatalf("Unable to update user: %v", err)
	}

	tools.PrintResource(t, newUser)
}
